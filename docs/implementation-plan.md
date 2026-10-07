# Bayesian calibration implementation plan

Public refactor status (2026-09-30): R1–R3 are complete and local R4 acceptance checks pass. An adequate production posterior/predictive comparison remains outstanding, so R4 is not marked complete as a production milestone. Section 15 records the implementation, interfaces, evidence and remaining scope. The following Stage 1–14 records describe historical work; removed checkpoint/job facilities are not public requirements.

Status: Stages 1–14a are complete. Stage 14a uses the user-supplied synthetic data with 4-unit slope matching, seed-1024 N(0,1²) field noise before matching/projection, and unspecified units. The selected offset and full slope-error diagnostic are recorded; local OLS accepts irregular, repeated, and unsorted per-curve depths and scans a regular offset grid. Stage 14b execution preparation is implemented for five methods × four chains, with unbounded sites and six production hours per chain. The pre-benchmark hot-path simplification and revised zero-covariance warmup policy are implemented and validated; Unity locked-environment setup, artifact verification, and the five-method 20-sweep execution check are complete. All five checks exited successfully, with warmup tuning concerns recorded below; the subsequent reduced-scale 100-sweep checks finished with five successes and two collapsed-NUTS numerical failures. MH startup variance 1e-6 is implemented. The whitening implementation passed 54 focused tests but is now deferred by the user; replacement eta-space diagonal/Kronecker/dense NUTS is implemented and validated (49 focused tests), using the approved partial-trace estimator and Gamma notation. The subsequent direct factor-moment accumulator and its checkpoint/integration regressions pass 35 tests. The authorized 21-way 200-sweep Unity check is finished: MH/MALA/MMALA and all nine uncollapsed-NUTS combinations completed, while all nine collapsed-NUTS combinations failed before the first mass update. The later MH/MALA/MMALA production attempt failed in all chains and is documented below. CV–NLPD and CV–WMSE length-scale selection are now available alongside profile likelihood; their library-only fits have been computed, but no new calibration run has used them. The plan consolidates D01–D10, R01–R04, U01–U02, and the current model and sampler decisions.

This is the single living plan. Keep decisions, milestone status, next task, and validation evidence here. Implement only the requested task and preserve unrelated user edits.

Current task (2026-10-07): the Section 16 symmetry/validation simplifications and selectable coefficient-GP SE, Matérn 3/2 and Matérn 5/2 kernels are implemented and locally validated. Matérn 3/2 is the default. The user explicitly selected radial standardized ARD Matérn for coefficient K_c only; spatial K_theta remains SE. Section 17 records the resolved model decision, implementation and passing reference/sampler/integration evidence. No kernel question remains unresolved. Next scientific work is adequate fresh posterior/predictive validation; Stage 14b and the production part of R4 remain incomplete.

Previous task (2026-09-30): the user-approved public refactor and local validation are complete. The method/experiment contract, fresh-run CLI, sampler JSONs, infrastructure removal, preparation separation and runtime failure context passed their local acceptance checks. Next scientific work is to predeclare adequate posterior/predictive comparisons and diagnose fresh full-protocol failures before deciding on any GP remedy or launching a production comparison. No production-budget or cluster run was launched. Stage 14b and the production-comparison part of R4 remain incomplete.

## 1. Scope and priorities

Build transparent Python/JAX research code for the specified Bayesian calibration posterior and compare collapsed MH, MALA, simplified MMALA, and uncollapsed NUTS. All compared algorithms must target the same joint posterior.

Public reproducibility requirement (user clarification, 2026-09-30): reproduce the model, sampling algorithms and experimental setup sufficiently to regenerate comparable results from a fresh run. Exact historical trajectories, failures, artifacts or environments are not the goal. Section 15 defines the revised acceptance criteria; historical validation records remain evidence of prior work rather than requirements for the public implementation.

Public implementation scope (user decision, 2026-09-30): remove checkpoint/restart, historical hash compatibility and general-purpose job orchestration rather than refactor them. Expose run-level options through the CLI; define sampler-specific options in configuration files with explicit CLI overrides. Earlier checkpoint/job architecture and validation records describe prior implementation work and are superseded for the public refactor by Section 15.

Priorities: statistical correctness, numerical correctness, testability, clarity, reproducibility, then performance. Use float64, explicit PRNG keys, documented shapes, and autodiff where appropriate. Develop on Apple Silicon with a CPU reference backend and support eventual Linux HPC runs. BlackJAX supplies NUTS transitions and standard Stan-style window adaptation; model densities and the outer sampling schedule remain explicit project code.

Implementation decision: prefer existing JAX, SciPy, or NumPy functions before adding helpers. Add a custom helper only when standard functions do not meet the requirement or when performance, numerical stability, autodiff, JIT, or readability provides a clear reason; briefly document that reason where the helper is defined.

## 2. Source specifications and review amendments

The source notes are located in:

`/Users/jaehoonkim/Library/Mobile Documents/iCloud~md~obsidian/Documents/PhD/Projects/Bayesian Model Calibration/`

| Note | Role |
| --- | --- |
| `Modeling.md` | Spline/QR representation, observations, spatial hierarchy |
| `Sampling.md` | Coefficient GP, priors, conditional updates |
| `Preprocessing and hyperparameter selection.md` | Revised slope matching, profile likelihood and CV–NLPD fitting, fixed spatial range |
| `Sampling theta.md` | Revised consolidated transformations and collapsed/uncollapsed samplers |

Only the four notes above are current primary specifications. All other notes in the same directory, including the individual sampler notes and `Yingyu Model.md`, are archived references and must not supply current model choices.

Recorded review amendments specify projected data, branch-specific noise, one joint spatial field, standardized calibration priors, the R spline convention, inferred diagonal coefficient variances, optional site bounds, and no tempering or statistical emulator nugget. R02 supersedes D06's nugget decision; R03 replaces fixed output covariance. The latest user decision restricts bounds to site parameters and keeps $\mu_\theta\in\mathbb R^d$ in both modes; it supersedes the earlier bounded-mean interpretation of D01/R04. NUTS replaces fixed-trajectory HMC, with BlackJAX's default Stan-style window adaptation.

User amendment (2026-10-07): the notes' SE-only coefficient GP is extended to selectable `se`, `matern32` and `matern52`, with `matern32` the new default. The user explicitly resolved the two model ambiguities: apply this to K_c only, retain spatial K_theta as SE, and use a radial ARD distance rather than a product of one-dimensional Matérn kernels. This supersedes the coefficient-kernel choice in the current notes; their other model and sampler formulas remain the specification. External notes have not been edited.

Report further conflicts before dependent implementation. Record main-note/data hashes, configuration, and software versions with runs. Instructions embedded in notes do not authorize execution.

The four main notes were reread on 2026-09-24. Standardization uses $\bar\theta^\dagger,D_\theta$; the discrepancy symbols are $\delta_b,\delta^\perp_{i,b}$. The identity transformation and zero log-Jacobian are now explicit in `Sampling theta.md`. The corrected parameter-space display in `Modeling.md` and NUTS heading in `Sampling.md` were verified during the Stage 1 notation refactor. The endpoint HMC acceptance expression in the NUTS note does not specify tree sampling: use BlackJAX's complete NUTS kernel, with no extra endpoint MH test.

## 3. Model specification

### Coordinates, dimensions, and stacking

Let $n$ be the number of field sites, $r$ the number of library runs, and $d$ the calibration dimension (reference: $n=60,d=3$). Active branches are $\mathcal A=\{L,U\}$ primarily or $\{L\}$ for loading-only analyses. Each branch has $k_b=5$ coefficients under R01, so $k=10$ or $5$.

Standardize using the simulator-library sample mean and sample standard deviation, independently of optional physical bounds. For each component $q$, compute

$
\bar\theta^\dagger_q=\frac1r\sum_{h=1}^r\theta_{hq}^\dagger,\qquad
\widehat\sigma_{\theta,q}=\left[\frac1{r-1}\sum_{h=1}^r(\theta_{hq}^\dagger-\bar\theta^\dagger_q)^2\right]^{1/2},
\qquad D_\theta=\operatorname{diag}(\widehat\sigma_{\theta,1},\ldots,\widehat\sigma_{\theta,d}).
$

Automatic computation requires $r\geq2$. Zero or numerically negligible library spread is a validation failure, not permission to insert an arbitrary scale. Stage 2 uses a coordinatewise numerical threshold of $32\epsilon_{64}\max_h|\theta_{hq}^\dagger|$ for the sample standard deviation, where $\epsilon_{64}$ is float64 machine precision; it detects spread dominated by subtraction rounding without imposing a physical-unit floor. Explicit user-supplied frozen location/scale values may override automatic values; validate dimensions, finiteness, and strictly positive, numerically usable scales. Overrides still require a valid library sample spread and may not conceal invalid library variation. Freeze the selected map before emulator fitting and calibration:

$
\widetilde\theta_i=D_\theta^{-1}(\theta_i-\bar\theta^\dagger),\qquad
\theta_i=\bar\theta^\dagger+D_\theta\widetilde\theta_i.
$

Apply the same map to library inputs, field parameters, physical bounds, and the spatial-mean prior center. Do not recompute it from field values or during MCMC. Standardized values are not assumed to lie in $(0,1)^d$.

If finite componentwise physical bounds $l,u$ are supplied, transform them to $\widetilde l=D_\theta^{-1}(l-\bar\theta^\dagger)$ and $\widetilde u=D_\theta^{-1}(u-\bar\theta^\dagger)$, and let $\mathcal S=\prod_q(\widetilde l_q,\widetilde u_q)$. Apply this support only to site parameters; the spatial mean $\mu_\theta$ always has support $\mathbb R^d$. If bounds are absent, $\mathcal S=\mathbb R^d$; do not substitute library extrema or other artificial bounds. The specified initial modes are a fully finite box or no bounds; mixed/one-sided transformations are not specified.

Let $\widetilde\Theta\in\mathbb R^{n\times d}$ have rows $\widetilde\theta_i^{\mathsf T}$, and let $\widetilde\theta_{1:n}$ denote its site-major stacked vector. Stack all coefficient vectors site/run first, then loading coefficients, then unloading coefficients:

$
c_f=\begin{pmatrix}c(\theta_1)\\\vdots\\c(\theta_n)\end{pmatrix}\in\mathbb R^{nk},
\qquad
c_s=\begin{pmatrix}c_1^\dagger\\\vdots\\c_r^\dagger\end{pmatrix}\in\mathbb R^{rk}.
$

The field coefficients are latent; fitted library coefficients are fixed observations. Matrix views $F_f\in\mathbb R^{n\times k}$ and $F_s\in\mathbb R^{r\times k}$ are reshapes, not separate random variables.

### Preprocessing and projected observations

Use the basis produced by `splines::bs(t, degree=3, knots=c(1/3,2/3), Boundary.knots=c(0,1), intercept=FALSE)`, without adding a separate intercept. It has five columns per branch. Match this convention for library and field curves; verify the Python basis against R reference values, including endpoints. The [R documentation](https://stat.ethz.ch/R-manual/R-devel/library/splines/html/bs.html) specifies dimension as the number of interior knots plus degree when the intercept is excluded.

Estimate the library depth offset $\hat h_0$ by loading-slope matching over a 4 nm window. Shift and rescale library curves as

$
h'=h-\hat h_0,\qquad L'=L(h)-L(\hat h_0),\qquad
t=\frac{h-\hat h_0}{h_{\max}-\hat h_0}.
$

Record the exact slope-fitting/search rule, branch orientation, grids, and load scaling in preprocessing configuration. Record physical-unit metadata when available; do not invent missing units. Preprocessing is fixed across sampler comparisons.

For each site and branch, compute a full-column-rank reduced QR factorization $\Phi_{i,b}=Q_{1,i,b}R_{i,b}$ and $\widetilde y_{i,b}=Q_{1,i,b}^{\mathsf T}y_{i,b}$. Stack branches and define

$
R_i=\operatorname{blockdiag}_{b\in\mathcal A}(R_{i,b}),\qquad
\Sigma_y=\operatorname{blockdiag}_{b\in\mathcal A}(\sigma_{y,b}^2I_5).
$

The projected model is

$
\widetilde y_i\mid c(\theta_i),\delta,\Sigma_y
\sim N_k\!\left(R_i[c(\theta_i)+\delta],\Sigma_y\right).
$

Let $R=\operatorname{blockdiag}(R_1,\ldots,R_n)$, $d_\delta=1_n\otimes\delta$, and $\Omega_y=I_n\otimes\Sigma_y$. Then

$
\widetilde y\mid c_f,\delta,\Sigma_y
\sim N_{nk}\!\left(R[c_f+d_\delta],\Omega_y\right).
$

Observation errors are independent across sites and branches. Orthogonal QR residuals are excluded from the posterior and noise updates. Loading-only mode has no unloading variance or unloading coefficient state.

### Exact library-conditioned GP and inferred coefficient variances

Use a selected unit-amplitude coefficient kernel in standardized coordinates. With $r_c(x,x')=[\sum_q (x_q-x'_q)^2/\lambda_q^2]^{1/2}$, the user-approved options (2026-10-07) are $K_c=\exp(-r_c^2/2)$ for `se`, $K_c=(1+\sqrt3 r_c)\exp(-\sqrt3 r_c)$ for `matern32` (default), and $K_c=(1+\sqrt5 r_c+5r_c^2/3)\exp(-\sqrt5 r_c)$ for `matern52`. These are radial ARD kernels with one length scale per calibration component; amplitude continues to be carried by inferred Sigma_c. The previous SE formula remains available:

$
K_c(x,x')=\exp\!\left[-\frac12\sum_{q=1}^d\frac{(x_q-x_q')^2}{\lambda_q^2}\right],
$

with Gram matrices

$
(K_{ff})_{ij}=K_c(\widetilde\theta_i,\widetilde\theta_j),\quad
(K_{fs})_{ih}=K_c(\widetilde\theta_i,\widetilde\theta_h^\dagger),\quad
(K_{ss})_{gh}=K_c(\widetilde\theta_g^\dagger,\widetilde\theta_h^\dagger),\quad
K_{sf}=K_{fs}^{\mathsf T}.
$

Their dimensions are $n\times n$, $n\times r$, and $r\times r$. There is **no statistical nugget** in these definitions. Fit the shared length-scale vector using the library-only profile marginal likelihood, the CV–NLPD criterion, or the later requested CV–WMSE diagnostic criterion defined below, then hold it fixed during calibration MCMC. The frozen Stage 14b prepared artifact used profile likelihood; changing the selection method requires a fresh prepared artifact and validation, not a mutation of existing checkpoints.

For each active branch,

$
\Sigma_{c,b}=\operatorname{diag}(\sigma_{c,b,1}^2,\ldots,\sigma_{c,b,5}^2),\qquad
\sigma_{c,b,j}^2\overset{\mathrm{iid}}{\sim}\operatorname{IG}(\alpha_{c,0,b},\beta_{c,0,b}),
\qquad \alpha_{c,0,b}=1.01,\quad \beta_{c,0,b}=0.01\ \text{by default}.
$

The user-specified branch priors `alpha_c_0` and `beta_c_0` are positive finite arrays of shape `(B,)`, in active-branch order. Each pair applies independently to all coefficient variances within that branch; defaults preserve the original common prior. Scalar overrides are rejected.

Combine branches into $\Sigma_c=\operatorname{blockdiag}_{b\in\mathcal A}(\Sigma_{c,b})$. Its diagonal entries, collected in `sigma_c2` (shape `(k,)`) and written $\sigma_{c,j}^2$ for $j=1,\ldots,k$ across active branches, are sampled during MCMC, not fitted and frozen with the length scales. The zero-mean separable joint GP is

$
\begin{pmatrix}c_f\\c_s\end{pmatrix}\mid\widetilde\Theta,\Sigma_c
\sim N_{(n+r)k}\!\left(0,
\begin{pmatrix}K_{ff}&K_{fs}\\K_{sf}&K_{ss}\end{pmatrix}\otimes\Sigma_c\right).
$

Exact conditioning gives

$
\mu_0=(K_{fs}K_{ss}^{-1}\otimes I_k)c_s,\qquad
K_0=K_{ff}-K_{fs}K_{ss}^{-1}K_{sf},\qquad
\Sigma_0=K_0\otimes\Sigma_c,
$

$
c_f\mid c_s,\widetilde\Theta,\Sigma_c\sim N_{nk}(\mu_0,\Sigma_0).
$

Because $\Sigma_c$ is inferred, the joint posterior includes both $p(c_s\mid\Sigma_c)$ and $p(c_f\mid c_s,\widetilde\Theta,\Sigma_c)$, multiplied by the inverse-Gamma variance priors. Fixed observed library coefficients still inform their unknown covariance. Omitting the library likelihood would change the posterior. It cancels in theta-only ratios, but not in coefficient-variance updates.

For library fitting, let $s_j=F_s[:,j]$ and $q_j(\lambda)=s_j^{\mathsf T}K_{ss}(\lambda)^{-1}s_j$. Under the specified zero GP mean, the coefficient-specific variance maximum-likelihood estimate at a candidate $\lambda$ is

$
\widehat{\sigma_{c,j}^2}(\lambda)=\frac{q_j(\lambda)}{r}.
$

Profile these variances out and maximize

$
\ell_{\mathrm{prof}}(\lambda)
=-\frac{k}{2}\log|K_{ss}(\lambda)|
-\frac{r}{2}\sum_{j=1}^k\log\!\left(\frac{q_j(\lambda)}r\right)+C,
$

where $C$ is constant in the length scales. Optimize $\xi_q=\log\lambda_q$ and set $\lambda_q=\exp(\xi_q)$. Sum over all coefficient columns of the active branches. Compute quadratic forms and log determinants by Cholesky solves. The divisor here is $r$, whereas input sample standardization uses $r-1$; do not center coefficient columns or estimate an additional GP mean.

The inverse-Gamma priors are not part of this profile-likelihood fitting objective. Profiled variances are nuisance fitting quantities only: calibration samples $\sigma_{c,j}^2$ under their specified priors rather than fixing them at $\widehat{\sigma_{c,j}^2}$. Record optimizer starts, tolerances, convergence diagnostics, and fitted length scales. A singular $K_{ss}$, or a zero column yielding $q_j=0$ and no positive interior variance optimum, must be reported rather than silently floored. Numerical safeguards follow the common jitter policy.

Alternative user-specified selection criterion (2026-09-29): for each held-out library run $j$, use the remaining $r-1$ rows of $F_s$ and the matching principal kernel matrix $C_{-j,-j}$. With zero GP mean, set $\widehat c_j=C_{j,-j}C_{-j,-j}^{-1}F_{-j}$, $\widehat\sigma_{c,\ell,-j}^2=F_{-j,\ell}^{\mathsf T}C_{-j,-j}^{-1}F_{-j,\ell}/(r-1)$, and $v_j=1-C_{j,-j}C_{-j,-j}^{-1}C_{-j,j}$. Minimize over log length scales the mean of $\tfrac12[k\log(2\pi)+\sum_\ell\log(v_j\widehat\sigma_{c,\ell,-j}^2)+\sum_\ell(c_{j,\ell}^\dagger-\widehat c_{j,\ell})^2/(v_j\widehat\sigma_{c,\ell,-j}^2)]$. This is the current source note's CV–NLPD definition. Factor each fold separately and use only its training rows for variance estimation; reject nonpositive or nonfinite predictive variances rather than floor them. The full-library $q_\ell/r$ in fit diagnostics remains a descriptive quantity for either method, never a fixed posterior parameter. The method changes only library length-scale selection; it does not replace the coefficient-variance priors or MCMC updates.

The user's subsequent WMSE criterion is exactly the final quadratic term of each fold's NLPD, averaged over held-out runs: $\operatorname{CV\!\text{-}WMSE}(\lambda_c)=r^{-1}\sum_j\sum_\ell(c_{j,\ell}^\dagger-\widehat c_{j,\ell})^2/(v_j\widehat\sigma_{c,\ell,-j}^2)$. Dividing by $k$ would only rescale it and leave its minimizer unchanged. It reuses the same fold predictions and variance definitions as CV–NLPD, but omits the logarithmic predictive-variance penalty. It is therefore not a proper predictive density score and can favor large predictive uncertainty. This selection choice affects only fixed $\lambda_c$ if used for a newly prepared experiment; it does not alter the GP model or posterior variance updates.

Numerical jitter is only a fixed, documented factorization safeguard when needed, not part of the statistical covariance. Check its effect by sensitivity tests and never adapt it within a transition or trajectory. Exact or persistent structural singularity must be reported as a modeling/implementation failure, not repaired by increasing jitter or silently excluding a neighborhood of parameter space.

Stage 3 defaults to zero jitter. A caller may supply one fixed nonnegative jitter for factorization and fitting; the unjittered $C_{ss}$ is checked first, with numerical singularity defined by $\lambda_{\min}\leq32\epsilon_{64}\lambda_{\max}$. Repeated library inputs, coincident field/library inputs, and singular unjittered field conditionals are rejected at the host validation boundary. Positive jitter produces an explicitly approximate factor-based evaluation and requires sensitivity checks. The JIT conditional kernel assumes validated locations; later targets must handle any invalid proposal without silently changing the covariance.

Stage 3b requires caller-supplied log-length-scale starts and explicit `gtol`, `ftol`, and `maxiter` settings. It uses SciPy L-BFGS-B with JAX value/gradient evaluations. `library_fit.method` selects `profile` (default, maximize log likelihood), `cv_nlpd` (minimize mean NLPD), or `cv_wmse` (minimize the mean foldwise variance-weighted squared error). Optional finite log bounds are an explicit optimizer search restriction, not a GP prior or model support. Record every start, success/failure message, objective, iteration/evaluation counts, SciPy version, fixed jitter, and profiled variances. Only fitted length scales are frozen for calibration; the profiled variances remain diagnostics and the actual $\sigma_{c,j}^2$ remain sampled parameters.

### Spatial hierarchy and standardized priors

Use one joint spatial covariance over all sites,

$
(C_\theta)_{ij}=\exp\!\left[-\frac{\|s_i-s_j\|^2}{2\rho_\theta^2}\right],\qquad \rho_\theta=12\ \text{by default}.
$

The default corresponds to two grid spacings when spacing is 6 in the same coordinate units. Keep the range fixed for the initial comparison; record the coordinate units from the data when available.

Let $\Sigma_\theta$ be the standardized spatial covariance parameter and $V_{\theta,0}$ the standardized reference prior covariance of the mean. The joint spatial prior is

$
\begin{aligned}
p(\widetilde\theta_{1:n},\mu_\theta,\Sigma_\theta)
\propto{}&N_{nd}(\widetilde\theta_{1:n};1_n\otimes\mu_\theta,C_\theta\otimes\Sigma_\theta)\\
&\times N_d(\mu_\theta;m_{\theta,0},V_{\theta,0})
\operatorname{IW}_d(\Sigma_\theta;\nu_{\theta,0},S_{\theta,0})\\
&\times\mathbf 1_{\mathcal S^n}(\widetilde\theta_{1:n}),\qquad \mu_\theta\in\mathbb R^d.
\end{aligned}
$

With optional site bounds, retain D01's global joint restriction, now without a mean-support indicator; do not introduce a hyperparameter-dependent field truncation normalizer. Its reference factors are not its bounded marginal hyperpriors, and their Gaussian covariance is not the actual bounded-field covariance. Without bounds, the site indicator is one and the ordinary Gaussian hierarchy applies.

Reference $d=3$ defaults are

$
\nu_{\theta,0}=5,\qquad S_{\theta,0}=I_3,\qquad V_{\theta,0}=4I_3,
\qquad
m_{\theta,0}=D_\theta^{-1}\!\left[
\begin{pmatrix}37850\\24060\\0.071\end{pmatrix}-\bar\theta^\dagger\right].
$

The mean is converted from the recorded physical reference center. Covariance defaults are defined directly on standardized coordinates. Physical output maps back using $\bar\theta^\dagger,D_\theta$; the physical Gaussian covariance parameter is $D_\theta\Sigma_\theta D_\theta^{\mathsf T}$. Require compatible explicit configuration for other calibration dimensions.

Other priors are

$
\delta\sim N_k(0,10^{-6}I_k),\qquad
\sigma_{y,b}^2\sim\operatorname{IG}(\alpha_{y,0,b},\beta_{y,0,b}),\quad b\in\mathcal A,
\qquad \alpha_{y,0,b}=1.01,\quad\beta_{y,0,b}=0.01\ \text{by default},
$

with independent branch-variance priors. Use shape/scale $p(v)\propto v^{-a-1}\exp(-b/v)$, and inverse-Wishart $p(S)\propto |S|^{-(\nu_{\theta,0}+d+1)/2}\exp[-\operatorname{tr}(S_{\theta,0} S^{-1})/2]$. Explicit user hyperparameters override defaults.

### Computational coordinates and theta targets

For finite supplied bounds, use

$
\widetilde\theta_{iq}=\widetilde l_q+(\widetilde u_q-\widetilde l_q)\operatorname{sigmoid}(\eta_{iq}),
$

$
\log J_{\mathrm{std}}(\eta)=\sum_{i,q}\left[
\log(\widetilde u_q-\widetilde l_q)-\operatorname{softplus}(-\eta_{iq})-\operatorname{softplus}(\eta_{iq})\right].
$

Without bounds, use $\eta_{iq}=\widetilde\theta_{iq}$ and $\log J_{\mathrm{std}}=0$. Priors are defined on standardized theta, not independently on eta. Do not reuse the old unit-interval Jacobian or add the physical affine Jacobian to a standardized-density target.

Finite $\eta$ maps to the interior of a finite box; its endpoints have infinite inverse coordinates. The source note's closed box and the open computational map agree for continuous densities up to their measure-zero boundary. Do not clip sigmoid values or Jacobian tails.

Let $\psi=(\delta,\{\sigma_{y,b}^2\}_{b\in\mathcal A},\mu_\theta,\Sigma_\theta,\{\sigma_{c,j}^2\}_{j=1}^k)$, and

$
m=R(\mu_0+d_\delta),\qquad V=R\Sigma_0R^{\mathsf T}+\Omega_y,
\qquad
\ell_{\mathrm{sp}}=\log N_{nd}(\widetilde\theta_{1:n};1_n\otimes\mu_\theta,C_\theta\otimes\Sigma_\theta).
$

For supported states, up to terms constant during a theta-only update,

$
\ell_{\mathrm{coll}}=\log N_{nk}(\widetilde y;m,V)+\ell_{\mathrm{sp}}+\log J_{\mathrm{std}},
\qquad
\ell_{\mathrm{uncoll}}=\log N_{nk}(c_f;\mu_0,\Sigma_0)+\ell_{\mathrm{sp}}+\log J_{\mathrm{std}}.
$

Condition on current coefficient variances in both targets. Conditional on $c_f$, the observation likelihood is constant in theta. Keep full-joint and theta-only densities distinct: terms such as the library likelihood must remain in the former. Do not replace joint densities with products of site conditionals.

Stage 4 implements both theta-only densities above and full collapsed/uncollapsed joint densities. The full versions add the normalized library coefficient likelihood, discrepancy prior, branch noise inverse-Gamma priors, spatial-mean Gaussian prior, spatial-covariance inverse-Wishart prior, and coefficient-variance inverse-Gamma priors. The full uncollapsed version also includes the projected observation likelihood given $c_f$. All terms use current $\sigma_{c,j}^2$ and branch noise variances. JAX lacks inverse-Gamma and inverse-Wishart log densities in the installed environment, so `targets.py` implements their normalized source-note formulas with JAX special functions and Cholesky solves. The finite-box target omits only the global restriction normalizer, constant across model states. Nonfinite evaluations return $-\infty$ without changing covariance factors; gradients are used only on valid positive-definite states.

## 4. Consolidated review decisions

| ID | Current decision |
| --- | --- |
| D01 | Globally restricted joint spatial prior when bounds are supplied; otherwise the unbounded Gaussian hierarchy. Bounds apply only to sites; $\mu_\theta$ is always unbounded (latest user amendment to D01/R04). |
| D02 | Projected observations only; orthogonal residuals excluded from posterior and variance updates. |
| D03 | Both branches primarily, loading-only supported; independent branch-specific noise variances. |
| D04 | One joint spatial fit across all sites. |
| D05 | Unit-amplitude ARD squared-exponential emulator; select library-only length scales by profile likelihood (U02) or the later user-specified CV–NLPD/CV–WMSE options, then freeze them in MCMC. Spatial squared-exponential range defaults to 12. Output variances are inferred under R03. |
| D06 | Superseded by R02: deterministic library and exact GP conditioning, with no statistical nugget. |
| D07 | Eta-space covariance preconditioner $P_i=\widehat S_i+\operatorname{tr}(\widehat S_i)I_d/(10^3d)$, multiplied by the sampler's proposal scale. |
| D08 | Fixed slope-matching preprocessing and the five-column R spline convention specified by R01. |
| D09 | Configurable standardized spatial priors and noise/discrepancy defaults in Section 3; R03–R04 supply coefficient priors and prior centers. |
| D10 | No tempering in the initial implementation/comparison. |
| R01 | Resolved: `splines::bs`, cubic, interior knots 1/3 and 2/3, boundary knots 0 and 1, `intercept=FALSE`; five coefficients per branch. |
| R02 | Resolved: no statistical nugget. Fixed documented numerical jitter only when necessary; sensitivity checks required; no structural repair by increasing jitter. |
| R03 | Resolved: diagonal coefficient covariance with independent inverse-Gamma variance priors, defaults $\alpha_{c,0}=1.01,\beta_{c,0}=0.01$; sample variances during MCMC. |
| R04 | Resolved: zero discrepancy mean; physical spatial-mean center $(37850,24060,0.071)^{\mathsf T}$ standardized using the library; optional user bounds on site parameters only, unbounded spatial mean, no invented bounds or units. |
| U01 | Resolved: library sample mean and sample standard deviation with denominator $r-1$; freeze before fitting/calibration. Fail on zero/negligible spread; explicit validated frozen-map overrides are allowed. |
| U02 | Resolved: profile marginal likelihood remains the default library-only length-scale selector, maximizing over log length scales after analytically profiling each diagonal output variance. The 2026-09-29 CV–NLPD option is an explicitly selected alternative from the current source note; the later CV–WMSE option uses only its quadratic term. Freeze fitted length scales, not output variances, during MCMC. |

The earlier reconciliation choices are resolved by the latest user decisions and corrected notes. Use only the four main notes; use unbounded `mu_theta`, optional site-only bounds, and BlackJAX NUTS with default Stan-style window adaptation. Section 2 records remaining editorial remnants. Numerical tolerances, optimizer starts, and run-specific preprocessing settings still need explicit configuration and validation; they must not silently change the specified model. Record physical-unit metadata when available without inventing missing values. Support for mixed or one-sided bounds remains outside the specified initial modes.

## 5. Implementation architecture

Grow the existing `src/bayesiancalibration/` package by milestone:

```text
state.py, data.py       # containers, preprocessing, frozen standardization
transforms.py          # physical / standardized / bounded or identity maps
linalg.py, gp.py        # Gaussian algebra, library fitting, exact conditioning
targets.py, gibbs.py    # explicit densities and conditional updates
samplers/
    metropolis.py      # random walk and preconditioned MALA
    mmala.py           # position-dependent metric
    nuts.py            # thin BlackJAX NUTS adapter
adaptation.py, mcmc.py  # warmup and explicit outer schedules
diagnostics.py, run.py  # reporting, configuration, checkpoints
```

Keep numerical kernels pure where practical and orchestration readable. Separate target construction from proposal mechanics, draws, and adaptation. Retain dense reference calculations after optimization. Avoid competing root-level implementations, generic class hierarchies, and a universal caching framework. Keep `main.py` thin when implementation is authorized.

This is the milestone-era module map. There is currently no package `diagnostics.py`; `run.py` implements checkpoints and `comparison.py` implements experiment execution. Section 15 governs the public refactor: checkpoint/restart, historical hash compatibility and general-purpose job orchestration are to be removed, while a thin fresh-run CLI and scientific experiment recipe are retained. The already deleted root-level legacy files are unrelated user edits and must not be restored as part of this review.

### Source notation in code

Use the source symbols for model-specific arguments, state fields, local quantities, and tests whenever practical. Transcribe them into readable ASCII (`theta_tilde`, `m_f_given_s`), and document the source symbol, conditioning, shape, and stacking. Keep generic names in genuinely reusable numerical helpers. A symbol must identify the same quantity, not merely a similar type of array.

| Revised-note code name | Meaning / earlier notation in this plan |
| --- | --- |
| `c_f`, `c_s` | Stacked field and library coefficients, shapes `(n*k,)` and `(r*k,)` |
| `C_ff`, `C_fs`, `C_ss` | Input Gram matrices, previously $K_{ff},K_{fs},K_{ss}$ |
| `C_f_given_s`, `m_f_given_s` | Library-conditioned input covariance $(n,n)$ and coefficient mean $(nk,)$, previously $K_0,\mu_0$ |
| `Sigma_f_given_s` | Derived alias for the full coefficient covariance $C_{f\mid s}\otimes\Sigma_c$, shape $(nk,nk)$; earlier $\Sigma_0$ |
| `m_y`, `V_y` | Collapsed projected-observation moments, previously $m,V$ |
| `m_f`, `V_f` | Conditional moments of $c_f$ after observing projected data |
| `m_delta`, `V_delta` | Conditional discrepancy moments in `gibbs.py`, with prior moments held in `CalibrationTarget.m_delta_0` (k,) and `CalibrationTarget.V_delta_0` (k,k) |
| `mu_theta` | Unbounded standardized spatial mean parameter $\mu_\theta$ |
| `m_theta`, `V_theta` | Conditional moments of `mu_theta`; distinct from its prior moments `m_theta_0`, `V_theta_0` |
| `alpha_y`, `beta_y` | Conditional branch-noise shape/scale in `gibbs.py`; its prior arguments and target fields are `alpha_y_0`, `beta_y_0`, both (B,) |
| `nu_theta`, `S_theta` | Conditional inverse-Wishart df/scale, distinct from prior `nu_theta_0`, `S_theta_0` and sampled covariance `Sigma_theta` |
| `sigma_c2` | GP coefficient-variance vector, shape `(k,)`, with $\Sigma_c=\operatorname{diag}(\texttt{sigma_c2})$ |
| `alpha_c`, `beta_c` | Conditional coefficient-variance shape/scale vectors, distinct from branch-specific prior vectors `alpha_c_0`, `beta_c_0` of shape `(B,)` |
| `delta`, `lambda_c`, `lambda_theta` | Discrepancy coefficients and emulator/spatial length scales (earlier length-scale notation: $\lambda,\rho_\theta$) |
| `theta_bar_dagger`, `D_theta` | Frozen library sample mean and diagonal sample-scale matrix |

Stage 1 uses these names in model-specific arguments, locals, and test fixtures. `m_f_given_s` is distinct from `m_theta`, and `Sigma_f_given_s` is distinct from the input covariance `C_f_given_s`. Stacked `d_delta` and `Omega_y` represent $1_n\otimes\delta$ and $I_n\otimes\Sigma_y$, not `delta` and `Sigma_y`. The module documents all shapes and derived aliases, including `Sigma_fy = Sigma_f_given_s @ R.T`, `L_y` as the Cholesky factor of `V_y`, and `m_y_given_cf = R @ (c_f + d_delta)`. Stage 1's generic Gaussian and solve helpers were later replaced by standard JAX functions. Callers using the old model-specific keyword names must migrate to the new names.

## 6. State and cache dependencies

| Container | Contents |
| --- | --- |
| Data | Projected observations `[n,k]`, QR factors `[n,k,k]`, locations, standardized library inputs `[r,d]`, library coefficient view `[r,k]`, branch slices |
| Fixed specification | Frozen library mean/sample scales (or explicit override and its provenance), optional bounds, priors, preprocessing metadata, fitted length scales and fitting diagnostics, spatial range, numerical-jitter policy |
| Model state | `eta[n,d]`, coefficient view `[n,k]`, discrepancy `[k]`, branch noise variances, coefficient variances `[k]`, standardized spatial mean `[d]`, standardized spatial covariance `[d,d]` |
| Sampler / warmup | Proposal tuning and NUTS state; separate adaptation accumulators/counters |
| Diagnostics / run | Transition statistics and failures; PRNG state, iteration, phase, hashes, versions, configuration |

Eta denotes bounded coordinates or the identity standardized coordinates, according to configuration. Derive standardized/physical site values without independently mutable copies. The spatial mean `mu_theta[d]` is a separate, always unbounded standardized variable; do not transform or clip it to the site bounds. Construct $\Sigma_c$ from current coefficient variances; do not store a conflicting fixed copy.

Prepare the run in order: validate/preprocess library data, compute or validate its frozen standardization, fit length scales, then construct fixed factors and MCMC caches. During fitting, each candidate length-scale vector changes $K_{ss}$, its factor, quadratic forms, and profiled variances. After fitting, precompute QR projections, fixed input/spatial covariance factors, branchwise sufficient products, $K_{ss}^{-1}F_s$, spatial precision information, and library column quadratic forms. Changing library data or the standardization override requires refitting and rebuilding these caches. Do not precompute $\Sigma_c$, its factor, or $R_i\Sigma_cR_j^{\mathsf T}$ as fixed quantities, or substitute profiled variances for sampled state.

| Change | Invalidation |
| --- | --- |
| One site | Emulator mean block, $K_0$ row/column and factor, collapsed covariance dependencies, spatial-prior dependencies |
| Coefficient variance | $\Sigma_c$, $\Sigma_0$, collapsed covariance/factor, coefficient conditional, relevant gradients/metrics; not $K_0$ or $\mu_0$ |
| Branch noise | $\Omega_y$, weighted sums, collapsed/conditional factors |
| Discrepancy | Collapsed mean and observation residuals |
| Spatial hyperparameters | Prior values, gradients, conditionals, and prior metric contribution |
| Field coefficients | Uncollapsed target, coefficient-variance residual quadratics, discrepancy/noise updates |

A leave-one-out factorization is reusable within one site proposal, not generally after another site or covariance parameter changes. Rejection preserves current caches. Recompute cached BlackJAX density/gradient whenever conditioning variables change.

## 7. MCMC execution and conditional updates

Use this common outer schedule:

1. Update discrepancy given current field coefficients and branch variances.
2. Update each active branch noise variance.
3. Update standardized spatial mean `mu_theta` from its Gaussian conditional in both site-bounds modes.
4. Update standardized spatial covariance.
5. Update coefficient-specific GP variances using current field coefficients and library data.
6. Update site parameters with the selected collapsed or uncollapsed kernel.
7. Draw field coefficients exactly conditional on the updated state.
8. Record the completed state.

For discrepancy, with configured prior $m_{\delta,0},V_{\delta,0}$ (zero and $10^{-6}I_k$ by default),

$
\Lambda_\delta=V_{\delta,0}^{-1}+\sum_iR_i^{\mathsf T}\Sigma_y^{-1}R_i,\qquad
h_\delta=V_{\delta,0}^{-1}m_{\delta,0}+\sum_iR_i^{\mathsf T}\Sigma_y^{-1}(\widetilde y_i-R_ic(\theta_i)).
$

The source-note conditional moments are $m_\delta=\Lambda_\delta^{-1}h_\delta$ and $V_\delta=\Lambda_\delta^{-1}$; draw from $N_k(m_\delta,V_\delta)$.

Stage 6a implements these moments and an explicit-key Gaussian draw in `gibbs.py`. It forms the stacked discrepancy design $H_\delta=R(1_n\otimes I_k)$, whitens it and $\widetilde y-Rc_f$ with the noise Cholesky factor, and assembles the source-note precision/information terms. Standard Cholesky/triangular solves provide the moments, including the covariance needed by JAX's standard Gaussian sampler. The pure numerical kernels support float64 JIT/vmap; the checked host `update_discrepancy` validates current coefficients and active branch variances and reports nonfinite numerical draws without jitter or repair. Prior arguments and target fields use the source-note names `m_delta_0`, `V_delta_0`. Current coefficients and noise are supplied every call, with no changing-state cache. This is the conditional with $c_f$ held fixed in both site-bounds modes, as required by the outer schedule. Stage 8 integrates these kernels into the complete outer sweep.

For $e_{i,b}=\widetilde y_{i,b}-R_{i,b}[c_b(\theta_i)+\delta_b]$,

$
\sigma_{y,b}^2\mid-\sim\operatorname{IG}\!\left(\alpha_{y,0,b}+\frac{nk_b}{2},\ \beta_{y,0,b}+\frac12\sum_i\|e_{i,b}\|^2\right).
$

Stage 6b implements independent active-branch parameter/draw kernels and checked `update_branch_noise`, using only these projected residuals. The tuple `branch_sizes` is static for JIT and loading-only returns one variance. JAX has no inverse-Gamma sampler: the documented `sample_inverse_gamma` transforms the standard unit-scale [JAX log-Gamma draw](https://docs.jax.dev/en/latest/_autosummary/jax.random.loggamma.html) as $v=\exp(\log b-\log G)$, $G\sim\operatorname{Gamma}(a,1)$. The log representation avoids premature underflow in $G$; there is no flooring or clipping, and host wrappers report unusable numerical draws. The same distribution primitive is reused for Stage 6e.

Let $w=C_\theta^{-1}1_n$, $a=1_n^{\mathsf T}w$, and

$
V_\theta=(V_{\theta,0}^{-1}+a\Sigma_\theta^{-1})^{-1},\qquad
m_\theta=V_\theta(V_{\theta,0}^{-1}m_{\theta,0}+\Sigma_\theta^{-1}\widetilde\Theta^{\mathsf T}w).
$

Draw $\mu_\theta\mid-\sim N_d(m_\theta,V_\theta)$ without truncation in either mode. Site bounds do not constrain this draw. This Gaussian conditional and the inverse-Wishart update below follow from the globally restricted joint prior; replacing it with a separately normalized truncated field hierarchy would change both updates.

Stage 6c implements these moments and a standard JAX Gaussian draw with checked `update_spatial_mean`. It uses the full fixed `C_theta` and current standardized sites/`Sigma_theta`; no changing-state moment is cached. The shared host coordinate checker documents its readability/validation role across the three spatial/GP wrappers. Conditional kernels remain pure JAX float64 functions with explicit keys, separate from host validation.

With $E_\theta=\widetilde\Theta-1_n\mu_\theta^{\mathsf T}$,

$
\Sigma_\theta\mid-\sim\operatorname{IW}_d\!\left(\nu_{\theta,0}+n,\ S_{\theta,0}+E_\theta^{\mathsf T}C_\theta^{-1}E_\theta\right).
$

Stage 6d computes this standardized sufficient statistic by whitening `E_theta` with a Cholesky triangular solve, then calls the explicit-key inverse-Wishart sampler. JAX has no IW sampler, so a small Bartlett construction is necessary for pure JAX JIT/vmap support: independent standard normal entries and chi-square diagonal draws form lower-triangular $A$ with $AA^{\mathsf T}\sim W_d(\nu,I_d)$. For $S=LL^{\mathsf T}$, solve $T=A^{-1}L^{\mathsf T}$ and return $T^{\mathsf T}T$, with no explicit matrix inverse. Its precision has law $W_d(\nu,S^{-1})$, consistent with the [SciPy Wishart relationship](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wishart.html) and [inverse-Wishart parameterization](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invwishart.html). The checked `update_spatial_covariance` accepts unbounded means in both site-bounds modes and reports nonfinite/nonpositive-definite draws without repairs or redraws. Real degrees of freedom $\nu>d-1$ are supported.

For coefficient column $j$ in branch $b(j)$, let $s_j=F_s[:,j]$, $f_j=F_f[:,j]$, and $e_{c,j}=f_j-K_{fs}K_{ss}^{-1}s_j$. For nonsingular exact GP factors, the full conditional implied by the joint GP is

$
\sigma_{c,j}^2\mid-\sim\operatorname{IG}\!\left(
\alpha_{c,0,b(j)}+\frac{r+n}{2},\quad
\beta_{c,0,b(j)}+\frac12\left[s_j^{\mathsf T}K_{ss}^{-1}s_j+e_{c,j}^{\mathsf T}K_0^{-1}e_{c,j}\right]\right).
$

The $r$-observation library contribution is required. Check this against the full joint log density; do not use the inverse-Gamma prior unchanged as though library conditioning removed all variance information. Length scales remain fixed during these updates.

Stage 6e implements coefficient-variance parameter/draw kernels and checked `update_coefficient_variances`. Library `q_s` supplies the first quadratic; current field residuals are whitened using current `C_f_given_s`. The shape for column `j` in branch `b` is `alpha_c_0[b] + (r+n)/2`, and its scale starts with `beta_c_0[b]`. Standard `jnp.repeat` expands the `(B,)` priors according to static `branch_sizes`; every active coefficient retains its own scale/draw. Full joint targets use the same branch expansion. The existing GP API is called with unit output variances only to retrieve its variance-independent mean and input conditional covariance; these are recomputed after current sites change. Fixed library factors/quadratics may be reused, while field residuals/covariances are never cached. The wrapper rejects structural singularity before sampling, including field/library coincidence under positive fixed jitter. Positive configured library jitter keeps the previously documented approximate factor-based interpretation, checked against the matching full joint GP. No numerical repair, new dependency, state mutation, or outer sweep is introduced by Stage 6.

For collapsed theta, integrate out all $c_f$ and hold current variances fixed. Follow the theta sweep immediately with the exact coefficient refresh before any coefficient-dependent update, including the variance update. A marginal-invariant theta transition plus exact refresh preserves the joint posterior; one valid MH transition per site suffices. An arbitrary single coefficient MCMC step is not a substitute without a separate proof.

For uncollapsed NUTS, hold $c_f$ and all other conditioning variables fixed throughout the entire tree construction and selection. Sequential site methods use latest accepted neighbors, not parallel stale conditionals. Do not record a collapsed-theta/old-coefficient intermediate pair as a complete draw. Local collapsed likelihood ratios may omit the complement only while it is constant in the candidate site value.

## 8. Numerical, autodiff, RNG, and adaptation strategy

Use float64 factorizations and solves; QR/SVD for spline least squares; rank checks before projection. Exact GP covariance matrices contain no statistical nugget. Numerical jitter perturbs the calculation even when labeled a safeguard: record its magnitude and placement, apply consistent policies across comparisons, test sensitivity toward zero, and do not describe a materially perturbed target as exact. Persistent structural degeneracy requires explicit failure, not silent repair or a parameter-dependent exclusion region.

For separable uncollapsed calculations, let $M_0$ reshape $\mu_0$, $E_c=F_f-M_0$, $K_0=L_KL_K^{\mathsf T}$, and $\Sigma_c=L_cL_c^{\mathsf T}$. Then

$
\log p(c_f\mid c_s,\widetilde\Theta,\Sigma_c)
=-\frac{k}{2}\log|K_0|-\frac{n}{2}\log|\Sigma_c|
-\frac12\|L_K^{-1}E_cL_c^{-\mathsf T}\|_F^2+\text{constant}.
$

The diagonal $L_c$ changes with coefficient variances. Varying $R_i$ generally prevents a simple Kronecker factorization of collapsed $V$; retain a dense reference. For an exact coefficient conditional draw under the specified covariance, independently draw $u\sim N(\mu_0,\Sigma_0)$ and $e\sim N(0,\Omega_y)$, then set

$
y_0=\widetilde y-Rd_\delta,\qquad
c_f=u+\Sigma_0R^{\mathsf T}V^{-1}(y_0-Ru-e).
$

Validate against analytic conditional moments. All displayed inverses denote solves in implementation; singular prior factors must be detected rather than hidden by this representation.

Stage 5 implements this correction in `gibbs.py` using JAX `random.multivariate_normal` with Cholesky factors and independent split keys, plus `cho_solve` for $V$. It is equivalent to the precision-form conditional in `Sampling.md` for arbitrary varying $R_i$; it does not use the common-design shortcut mentioned in that note's footnote. The custom model-specific correction avoids factoring a posterior covariance formed by subtraction; Gaussian generation and linear solves use standard functions. The pure numerical kernel supports JIT/vmap and assumes valid float64 positive-definite inputs. The checked host entry point recomputes GP and observation moments from current `eta`, `delta`, branch noise, and `sigma_c2`, rejects singular unjittered field factors even with configured numerical jitter, and reports nonfinite draws. It adds no jitter or covariance repair. A caller supplies a fresh key per update; no changing-state moments or keys are stored internally. A positive configured library jitter retains Stage 3's explicitly approximate interpretation. Stage 8 integrates the exact refresh immediately after the collapsed theta sweep.

Use JAX `value_and_grad` through kernel conditioning and Cholesky solves. Use forward-mode site Jacobians for the MMALA conditional mean/covariance; retain analytic derivative checks. For finite bounds,

$
D_i=\operatorname{diag}\!\left((\widetilde u_q-\widetilde l_q)s_{iq}(1-s_{iq})\right),
\qquad s_{iq}=\operatorname{sigmoid}(\eta_{iq}).
$

Without bounds, $D_i=I_d$, the log-Jacobian gradient and curvature vanish, and the prior gradient/metric uses identity transformation. The bounded Jacobian curvature is $\operatorname{diag}(2s_{iq}(1-s_{iq}))$. Do not substitute standardized theta itself for $s_{iq}$ under arbitrary bounds.

Distinguish MALA covariance preconditioners from MMALA precision-like metrics. The simplified metric is a positive-definite surrogate with exact forward/reverse proposal correction, not a reason to add a metric-volume factor to the target. Validate NUTS's complete target, inverse-mass convention, and conditioning refresh; delegate tree construction, termination, and proposal selection to BlackJAX. Test transformation tails without clipping.

Start with readable orchestration and pure numerical kernels. JIT complete transitions after reference tests pass, then fixed-size sweep chunks. Keep preprocessing, static validation, I/O and reporting outside JIT; use small checkified device predicates to surface dynamic numerical failures. Pass changing conditioning arrays explicitly. Use BlackJAX NUTS; keep tuned parameters separate from cached density/gradient.

Split explicit PRNG keys by chain and update; use separate proposal/acceptance subkeys. Save keys, model/sampler/warmup state, configuration, standardization, hashes, versions, dtype/backend, and numerical policy. Cross-platform agreement need not be bitwise.

Adapt history-dependent tuning only during warmup; freeze it for retained sampling. D07's ridge vanishes at zero empirical variance, so initialize from a declared positive-definite covariance and retain it until estimation is usable. Include repeated rejected states. Freeze tuning through each forward/reverse calculation. Initially tune MMALA step size with a declared metric ridge. NUTS uses the window adaptation specified below. Position-dependent MMALA geometry is part of the fixed production kernel; acceptance targets remain heuristics.

### BlackJAX NUTS and window adaptation

Use the standard `blackjax.window_adaptation(blackjax.nuts, logdensity_fn)` defaults as the reference. The installed BlackJAX 1.6.2 default uses diagonal mass adaptation, identity initialization, initial step size 1.0, target acceptance 0.8, and 1,000 warmup iterations by default. Its default schedule has 75 initial fast iterations, expanding slow windows starting at 25, and 50 final fast iterations; retain its built-in short-warmup handling and regularization. Freeze the final step size and inverse mass matrix for retained sampling. Record the resolved defaults, actual warmup length, package version, and NUTS limits in every run; `uv.lock` also contains a Python-dependent 1.3 resolution, so a shared lockfile alone does not establish identical defaults across environments. See the [BlackJAX window-adaptation reference](https://blackjax-devs.github.io/blackjax/autoapi/blackjax/adaptation/window_adaptation/index.html).

Integrate the library's standard schedule and adaptation updates into the outer Gibbs warmup: one NUTS update of all `eta[n,d]` per sweep, then refresh coefficients. Feed the selected eta position and NUTS acceptance statistic to adaptation once per sweep. Rebuild the NUTS log density/gradient at the current position after conditioning variables change; preserve the adaptation state across sweeps. Do not restart a complete window warmup each sweep or silently adapt against one frozen initial conditional. The high-level driver takes a fixed log density; the Gibbs adapter must reuse the corresponding library adaptation components with unchanged defaults, and match the high-level driver on a fixed-target reference test. This is window adaptation embedded in Gibbs, not a claim that the changing-conditional warmup is a stationary NUTS chain.

Use BlackJAX's `inverse_mass_matrix` convention directly; do not manually invert a returned tuning parameter. Retain default NUTS integration and stopping settings, record divergences, integration counts and tree-limit hits, and freeze adaptation before retained draws. A default warmup length is a baseline, not evidence of convergence.

Stages 11–12 implement this protocol with `initialize_nuts_warmup`, `run_nuts_warmup`, and `run_fixed_nuts`. `collapsed=False` selects the uncollapsed current-c_f conditional; `collapsed=True` integrates c_f. Both use all-site blocking and identical resolved adapter defaults. `nuts_sweep` calls standard NUTS initialization every outer sweep. Diagnostics include acceptance, final conditional density, divergence/turning flags, integration and expansion counts, the reached expansion cap, and the step size/diagonal inverse mass actually used for that transition. Standard window statistics persist across Gibbs conditionals and freeze at the final warmup boundary. The adapter uses the installed BlackJAX 1.6.2 staged engine, the same engine as its high-level window driver; future package changes require a new equivalence check.

### Simplified collapsed MMALA (Stage 13)

`collapsed_mmala_metric` computes the source note's conditional-observation Fisher matrix from both mean and covariance derivatives, plus transformed spatial conditional precision, coordinate-Jacobian curvature, and the declared ridge. `collapsed_mmala_sweep` recomputes this metric at the current and candidate eta_i for the exact normalized Gaussian proposal ratio. It visits sites sequentially with their latest neighbors and uses the complete joint theta target. `epsilon_G` is fixed tuning and affects the proposal only; it is not a GP nugget or a target-density term.

`initialize_mmala_warmup` requires `num_warmup`, initial `epsilon`, and positive `epsilon_G`. Standard dual averaging updates epsilon once per completed Gibbs sweep using mean site probabilities, with target acceptance 0.574 by default. `run_mmala_warmup` freezes the final averaged epsilon and DA state; `run_fixed_mmala` continues to evaluate G_i at each position. Covariance estimation is not part of MMALA warmup. These model-specific assembly/orchestration adapters are necessary because standard MALA does not supply this conditional Fisher metric or a position-dependent covariance; numerical factors, draws, densities, MH, and DA use standard library functions.

```python
key_u, key_c, key_m = jax.random.split(key, 3)
uncollapsed = initialize_nuts_warmup(target, model_state, key_u, collapsed=False)
collapsed = initialize_nuts_warmup(target, model_state, key_c, collapsed=True)
mmala = initialize_mmala_warmup(
    target, model_state, key_m, num_warmup=num_warmup,
    epsilon=initial_epsilon, epsilon_G=declared_metric_ridge,
)
collapsed, warmup_samples, warmup_info = run_nuts_warmup(target, collapsed)
collapsed, samples, diagnostics = run_fixed_nuts(target, collapsed, num_sweeps=100)
mmala, warmup_samples, warmup_info = run_mmala_warmup(target, mmala)
mmala, samples, diagnostics = run_fixed_mmala(target, mmala, num_sweeps=100)
```

Checkpoint schema 3 supports RW, MALA, both NUTS targets, and MMALA with separate sampler tags. It includes NUTS window schedules/defaults/limits or MMALA epsilon/ridge/DA state, and enforces tuning-payload compatibility. Existing schema/code hashes require a new run or explicit migration when implementation changes. The synthetic posterior checks establish identities/invariance on reference problems; Stage 14 must assess convergence and comparisons on real data.


### Fixed-proposal collapsed random walk (Stage 7)

`samplers/metropolis.py` implements `collapsed_random_walk_sweep` as a pure JAX site sweep and `update_collapsed_random_walk` as its checked host boundary. Supply `eta[n,d]`, current `delta[k]`, `sigma_y2[B]`, `mu_theta[d]`, `Sigma_theta[d,d]`, `sigma_c2[k]`, and explicit fixed SPD proposal covariances `V_prop[n,d,d]`. Each matrix is the complete eta-space covariance including any declared scale; no empirical adaptation or implicit regularization occurs in Stage 7. The returned eta and `RandomWalkSweepInfo` keep model position separate from per-site acceptance probabilities/decisions and the final density diagnostic.

The adapter uses the standard [BlackJAX random-walk MH kernel](https://blackjax.readthedocs.io/en/latest/_modules/blackjax/mcmc/random_walk.html) (validated locally with version 1.6.2) and JAX Gaussian draws. Custom code only selects the site block and supplies the calibration target/schedule. Sites are visited in order using independently split keys. Fixed Gaussian proposals are symmetric in eta, so the exact proposal ratio is one. The existing joint collapsed theta density supplies all likelihood, spatial-prior, and coordinate-Jacobian terms; no metric-volume term or clipping is introduced. Within a sweep, accepted positions/densities feed subsequent sites and rejected states repeat. Each new call initializes its density from the supplied current conditioning variables, so diagnostics are never reused as an inter-sweep density cache.

Host checks validate float64 mode, finite shapes, positive variances, symmetric SPD spatial/proposal covariances, finite initial target, and initial/final unjittered GP geometry. The pure kernel assumes valid initialization and rejects nonfinite candidate targets through the existing target's negative-infinity result. Structural or numerical failure in a final host-checked state is reported without jitter, repair, or redraws; continuous JIT proposals follow the existing GP kernel policy. After the collapsed theta sweep, the exact coefficient refresh remains required before coefficient-dependent updates. Stage 8 implements full outer-sweep/restart integration.


### Complete outer sweeps and checkpoint boundaries (Stage 8)

`CalibrationState` holds only changing model variables: `eta[n,d]`, site-major `c_f[n*k]`, `delta[k]`, `sigma_y2[B]`, `mu_theta[d]`, `Sigma_theta[d,d]`, and `sigma_c2[k]`. `RandomWalkChain` separately holds the model state, next unused scalar JAX key, next-sweep `V_prop[n,d,d]`, completed iteration count, phase, and optional `RandomWalkAdaptationState`. The chain container was named `FixedRandomWalkChain` in Stage 8; Stage 9 renames it to reflect both warmup and sampling. Direct fixed-tuning initialization remains supported with no adaptation state. `GibbsSweepInfo` keeps theta diagnostics separate and evaluates the full uncollapsed joint density after coefficient refresh.

`collapsed_gibbs_sweep` follows the specified seven-update schedule exactly, passing freshly updated values to each dependent update. It recomputes GP moments at the old sites for the coefficient-variance update and at the new sites for the exact coefficient refresh. Every sweep refreshes `c_f`, including sweeps where every theta proposal was rejected. The `split-8-v1` key protocol splits the incoming scalar key into the next unused key and seven distinct update keys in schedule order. Internal coefficient/MH draws retain their own established subkey splits. No changing-state density, gradient, or covariance cache crosses a sweep boundary.

`initialize_random_walk_chain` validates an explicitly supplied complete initial model state and proposal tuning. `run_fixed_random_walk` JIT-compiles the pure numerical sweep and uses readable host orchestration for positive-length chunks. It validates complete states before recording, includes rejected theta states in returned arrays, and advances the iteration/key only after a sweep succeeds. Samples and diagnostics have a leading sweep axis. Initialization/validation, I/O, and reporting stay outside JIT; invalid or nonfinite states and structural GP singularity stop the run without repair. The input chain is immutable, and intermediate sweep states are never returned.

`run.py` provides `save_checkpoint` and `load_checkpoint`. A single pickle-free compressed NumPy archive stores complete model arrays, tuning, raw key data plus its implementation and typed/legacy representation, iteration/phase, the entire fixed target including frozen standardization and numerical policy, array checksums, implementation hashes, numerical-package versions, relevant JAX/XLA settings, backend/dtype/Python/platform provenance, caller configuration, and caller-supplied source-note/data hashes. PRNG serialization uses standard [JAX key_data](https://docs.jax.dev/en/latest/_autosummary/jax.random.key_data.html) and [wrap_key_data](https://docs.jax.dev/en/latest/_autosummary/jax.random.wrap_key_data.html). Fixed library factors are fingerprinted with their dependencies; no changing-state density/gradient cache is stored. Schema 3 also stores the separate sampler-specific warmup schedules and standard covariance/dual-averaging/window accumulators when present. Real runs supply the four primary notes and input-data paths through `source_paths`; synthetic references may use an empty or explicitly synthetic provenance map.

Saving validates the boundary, writes/fsyncs a temporary archive in the destination directory, and replaces the checkpoint atomically. Failed replacement leaves the previous archive intact and cleans up the temporary file. Loading requires the explicitly supplied target to match every saved fixed quantity and validates the model, tuning, key, phase, iteration, schema, checksums, code, numerical-package versions, and JAX/XLA settings. Caller-provided `source_paths` can verify the current notes/data. Changed model inputs or implementations require an explicit new run or future migration support. Same-environment continuation is tested bitwise; backend/platform provenance is recorded without claiming cross-platform bitwise equivalence.

```python
chain = initialize_random_walk_chain(target, model_state, key, V_prop)
chain, samples, diagnostics = run_fixed_random_walk(target, chain, num_sweeps=100)
save_checkpoint("chain.npz", target, chain,
                configuration=run_configuration, source_paths=source_paths)
chain, metadata = load_checkpoint("chain.npz", target, source_paths=source_paths)
chain, more_samples, more_diagnostics = run_fixed_random_walk(
    target, chain, num_sweeps=100
)
```

The Stage 8 joint-stationarity reference draws the full scalar library-conditioned hierarchy independently with NumPy/SciPy, applies site bounds to the entire joint sample, weights by the integrated observation likelihood, and draws coefficients from the independently calculated Gaussian observation conditional. This preserves the globally restricted joint prior under bounds. Complete sweeps are compared with reference variable means, second moments, and cross moments, with both reference importance uncertainty and transition Monte Carlo uncertainty included. This validates the partial-collapse ordering and coefficient refresh jointly; it is not a convergence claim for arbitrary real-data runs.

### Random-walk warmup and frozen production (Stage 9)

`initialize_random_walk_warmup` requires explicit positive `num_warmup` and `num_initial`, with `num_initial <= num_warmup`. The user-approved (2026-09-28) default initial `V_prop` is `1e-6 I_d` per site, superseding the source note's identity startup; callers may supply a declared SPD `(n,d,d)` proposal. `RandomWalkAdaptationState` stores the lengths, initial tuning, and site-wise mean `(n,d)`, M2 `(n,d,d)`, and integer counts `(n,)` separately from model variables. The standard dense [BlackJAX Welford algorithm](https://blackjax-devs.github.io/blackjax/autoapi/blackjax/adaptation/mass_matrix/index.html) supplies online moments. Every completed warmup eta state contributes, including repeated rejections and states from the initial fixed period; the initialization position is excluded. Empirical covariance uses count minus one.

The first `num_initial` sweeps use the initial tuning. After a complete sweep, the estimate prepares tuning for the next sweep using exactly $V_{\mathrm{prop},i}=(2.38^2/d)[\widehat S_i+\operatorname{tr}(\widehat S_i)I_d/(1000d)]$ from D07. A site needs at least two completed states before estimating covariance. Once eligible, exactly zero empirical covariance retains the previous valid proposal temporarily and increments a site-level tuning-issue counter. A nonzero nonfinite/non-SPD estimate fails immediately. At the end of warmup, any site with exactly zero empirical covariance fails the warmup and prevents production (latest user decision; supersedes the earlier zero-scatter failure/hold rules). Positive-trace rank-deficient estimates receive only the prescribed ridge. No extra regularizer, clipping, redraw, or key consumption is introduced. As of the user-authorized 2026-10-07 simplification, stored M2 is normalized once; repeated sample-covariance averaging and triangle mirroring are removed. Covariance boundary checks permit negligible symmetry roundoff (rtol=1e-12, atol=1e-14) without modifying entries. If `num_initial == num_warmup`, initial tuning remains fixed; the final zero-scatter gate still applies before production.

`run_random_walk_warmup` runs a positive chunk no longer than the remaining warmup, or completes all remaining sweeps when the length is omitted. Returned warmup samples/diagnostics are separate from production. Adaptation runs only after the complete Gibbs sweep and immediate coefficient refresh, so each site's forward/reverse MH calculation uses fixed tuning. The final warmup sweep first requires nonzero empirical covariance at every site, then sets `phase='sampling'` and freezes the proposal and all accumulators. `run_fixed_random_walk` requires sampling phase and leaves them unchanged. The iteration counts both phases; accumulator counts stop at `num_warmup`.

Checkpoint schema 2 preserves partial warmup and the frozen boundary, validates schedule/count/phase consistency, and retains the existing target, code, and numerical-environment checks. Schema 1 has no automatic migration; it is rejected explicitly, as changed implementation hashes also preclude silently restarting an old run. Typed and legacy-key continuation is tested bitwise on the same environment. Finite warmup and positive proposal covariance do not establish real-data convergence; convergence assessment remains later work.

```python
chain = initialize_random_walk_warmup(
    target, model_state, key, num_warmup=1000, num_initial=100
)
chain, warmup_samples, warmup_diagnostics = run_random_walk_warmup(
    target, chain, num_sweeps=400
)
save_checkpoint("warmup.npz", target, chain,
                configuration=run_configuration, source_paths=source_paths)
chain, metadata = load_checkpoint("warmup.npz", target, source_paths=source_paths)
chain, remaining_warmup, remaining_diagnostics = run_random_walk_warmup(target, chain)
chain, samples, diagnostics = run_fixed_random_walk(target, chain, num_sweeps=100)
```

### Collapsed preconditioned MALA (Stage 10)

`collapsed_mala_sweep` and its checked host boundary `update_collapsed_mala` visit sites sequentially using the current collapsed joint theta target. Supply the same conditioning arrays as the random walk, SPD eta-space `V_prop[n,d,d]`, and a declared positive scalar `epsilon`. The source-note proposal is $N_d(\eta_i+\epsilon^2V_{\mathrm{prop},i}\nabla_i\ell_{\mathrm{coll}}/2,\epsilon^2V_{\mathrm{prop},i})$. `V_prop` is a covariance preconditioner, excluding epsilon-squared and the random-walk multiplier. The exact acceptance ratio includes both forward and reverse Gaussian proposal densities. Coordinate Jacobians already belong to the target; no metric-volume factor is added.

The installed BlackJAX 1.6.2 standard [MALA kernel](https://blackjax-devs.github.io/blackjax/autoapi/blackjax/mcmc/mala/index.html) supplies autodiff, isotropic diffusion, asymmetric acceptance, and safe rejection of nonfinite candidate energies. Its isotropic interface is adapted using the local affine map $\eta_i=\mathrm{origin}_i+L_i z$, $L_iL_i^{\mathsf T}=V_{\mathrm{prop},i}$, with `z=0` and `step_size=epsilon^2/2`. This gives the specified eta-space proposal by the chain rule; the constant affine determinant cancels between directions. Starting at local zero preserves rejected states exactly. Density and gradient are initialized afresh at every site from the latest other sites and conditioning variables, with no inter-site or inter-sweep gradient cache. Custom code only supplies this calibration-specific site map and schedule; it does not implement a new MH acceptance kernel.

The checked boundary validates the existing float64 shapes/support/unjittered GP geometry, SPD preconditioners, a finite positive representable diffusion scale, and finite initial/final collapsed gradients. Unusable tuning is rejected without clipping or repair. The pure transition supports JIT/vmap/scan and explicit keys; site-key splitting and BlackJAX's proposal/acceptance split match the established theta protocol.

`MALAChain` separates model variables, next unused key, preconditioners, scalar epsilon, counters/phase, optional Welford covariance state, and optional `MALAStepSizeAdaptationState`. `initialize_mala_chain` starts fixed production; `initialize_mala_warmup` uses the declared Stage 9 lengths and identity default/explicit SPD initial tuning. Reuse the standard Welford estimator and D07 ridge, including repeated rejections. MALA's empirical `V_prop` is $P_i=\widehat S_i+\operatorname{tr}(\widehat S_i)I_d/(1000d)$, with multiplier 1.0; epsilon-squared is applied in the transition. The same temporary exact-zero hold, terminal zero-scatter failure, site counters, initial covariance period, completed-sweep adaptation, and final freeze rules apply. The user's subsequent epsilon-adaptation request supersedes Stage 10's original fixed-epsilon warmup decision.

`initialize_mala_warmup(..., target_accept=...)` enables standard [BlackJAX dual averaging](https://blackjax-devs.github.io/blackjax/autoapi/blackjax/adaptation/step_size/index.html) with a target in (0,1), defaulting to 0.574 as requested by the user. Omitting the argument enables epsilon adaptation with that target; explicitly pass `target_accept=None` for fixed epsilon. Tune the common epsilon directly, using `dual_averaging_adaptation` defaults `t0=10`, `gamma=0.05`, `kappa=0.75`; the integrator still receives epsilon-squared/2. After each complete warmup Gibbs sweep, update once with the mean site acceptance probability (including rejected proposals), rather than the fraction of binary accept decisions. The next warmup sweep uses `exp(log_step_size)`; the final boundary freezes the standard `final()` averaged value. The existing initial fixed period holds covariance fixed while epsilon adaptation runs from the first sweep. Covariance and epsilon updates use the just-completed sweep with its unchanged forward/reverse tuning. No new windows, DA resets, clipping, or random draws are introduced.

Pass the current epsilon as a dynamic argument to the JIT sweep, so chunk execution cannot retain a captured initial scale. `MALAStepSizeAdaptationState` keeps target/initial epsilon and every standard scalar DA statistic separate from the covariance moments and model state. Its update count must match completed warmup; initial center/statistics and current/final epsilon must agree with configuration/phase. Production freezes both adaptation containers. Schema-2 checkpoints store the configuration and all DA scalars in the checksummed `step_size.*` payload, with the separate `mala-dual-averaging-v1` protocol. Invalid or unrepresentable scales stop the checked run without repairs or a partial returned chain. Joint covariance/scale adaptation in changing Gibbs conditionals is warmup tuning; a target acceptance probability is a heuristic and does not establish convergence.

`collapsed_gibbs_sweep(..., epsilon=epsilon)` selects MALA only for the theta update, preserving all seven update keys, current-input dependencies, and the immediate exact `c_f` refresh even after rejection. Omitting epsilon preserves random-walk behavior. `run_mala_warmup` returns only warmup history, and `run_fixed_mala` requires sampling phase and freezes epsilon/preconditioners/statistics. Both use the shared completed-sweep recording/failure logic. Sampler-tagged schema-2 checkpoints include scalar float64 `sampler.epsilon` in the checksummed payload, and preserve partial/frozen warmup statistics. Loading returns the matching chain type; sampler/driver mismatches are rejected.

```python
chain = initialize_mala_warmup(
    target, model_state, key, num_warmup=1000, num_initial=100,
    epsilon=0.3  # target_accept defaults to 0.574; epsilon adapts during warmup.
)
chain, warmup_samples, warmup_diagnostics = run_mala_warmup(target, chain)
chain, samples, diagnostics = run_fixed_mala(target, chain, num_sweeps=100)
save_checkpoint("mala.npz", target, chain,
                configuration=run_configuration, source_paths=source_paths)
chain, metadata = load_checkpoint("mala.npz", target, source_paths=source_paths)
```

## 9. Validation and fair comparison

| Test group | Acceptance evidence |
| --- | --- |
| Standardization/transforms | Sample mean and $r-1$ scales; override validation; failure on zero/negligible spread; frozen-map reuse and physical-center conversion; finite-box and identity-mode Jacobians |
| Library fitting | Profile objective equals joint library likelihood at $\widehat{\sigma_{c,j}^2}=q_j/r$ up to a constant; numerical variance maximization agrees; log-length-scale gradients, zero-column failures, and loading-only mode checked |
| Spline/QR/noise | Five-column basis matches R, including endpoints; rank/projection checks; branch-specific noise and loading-only mode |
| GP | Direct exact joint/conditional agreement; numerical-jitter sensitivity; structural-singularity detection at repeated/library inputs |
| Priors/conditionals | Joint-density ratios match unbounded Gaussian spatial mean in both site-bounds modes, inverse-Wishart, discrepancy, noise, and coefficient-variance conditionals |
| Coefficient variances | Both library and field quadratic terms and $(r+n)/2$ shape contribution; moments and cache invalidation |
| Targets/derivatives | Full and sitewise collapsed ratios agree; autodiff matches analytic/finite-difference checks in both bounds modes |
| Transitions/state | Proposal densities, metric positivity, NUTS reference-target behavior, default-window equivalence, conditioning refresh, frozen fitted length scales but sampled coefficient variances, frozen adaptation, restart |
| Posterior | Quadrature/known-target checks, partial-collapse joint covariance, collapsed/uncollapsed agreement |

With fixed library implicit and current coefficient variances included in $\psi$, test

$
\log p(\widetilde y,c_f\mid\widetilde\Theta,\psi)
=\log p(\widetilde y\mid\widetilde\Theta,\psi)
+\log p(c_f\mid\widetilde y,\widetilde\Theta,\psi).
$

Test the full joint GP factorization separately to ensure $p(c_s\mid\Sigma_c)$ is retained for inferred variances. Later simulation-based calibration must reproduce the actual joint model and library-conditioning protocol. Under bounds, do not generate from nominal hyperpriors followed by a separately normalized truncated field prior.

Compare the practical strategies first; add matched collapsed/uncollapsed NUTS or matched sitewise proposals to separate collapsing from blocking/proposal effects. Keep data, preprocessing, standardization, bounds mode, priors, fitted length scales, coefficient-variance updates, numerical policy, coefficient refresh, and outer schedule common. Use multiple chains/seeds and establish posterior agreement before performance claims.

Report convergence, bulk/tail ESS, Monte Carlo error, numerical/NUTS failures, and ESS per second for sites, spatial contrasts, both active noise variances, coefficient variances, discrepancy, covariance, and predictions. Time complete sweeps including covariance updates and coefficient refresh. Separate compilation/warmup/retained sampling and synchronize JAX timing. Tempering is excluded.

## 10. Staged roadmap

The specification includes U01–U02 and the latest main-note, site-only-bounds, and NUTS decisions; Section 2 records editorial remnants. Stages 1–14a passed their acceptance tests. Stage 14a uses synthetic data as requested; pilot comparisons at Stage 14b have not started.

| Stage | Deliverable | Acceptance evidence |
| --- | --- | --- |
| 1 — complete | Tiny projected Gaussian fixture | Five tests pass: dense SciPy agreement, conditional moments, joint/marginal/conditional identity, unequal branch noise, loading-only behavior, float64, and JIT compatibility |
| 2a — complete | Library standardization | Sample mean, $r-1$ scales, validated overrides, and zero/negligible-spread failure |
| 2b — complete | Coordinate maps and standardized prior | Frozen-map reuse, finite bounds, identity mode, prior-center and Jacobian checks |
| 3a — complete | Exact library conditioning with supplied length scales | Dense agreement, jitter sensitivity, singularity detection |
| 3b — complete | Library-only profile, CV–NLPD and CV–WMSE length-scale fitting | Analytic $q_j/r$ profile reference, foldwise NLPD/WMSE and gradient references, fit diagnostics, fixed length scales with sampled output variances |
| 4 — complete | Collapsed/uncollapsed targets with current coefficient variances | Density identities and gradients |
| 5 — complete | Exact coefficient refresh | Independent precision-form moments, 40,000 draws each for two-branch/loading-only empirical moments, RNG/JIT, current-state dependencies, and singularity rejection |
| 6a — complete | Discrepancy update | Independent Gaussian conditioning and one-site moments, full-joint ratios in both site-bounds modes, 40,000 draws per branch mode, RNG/JIT, current-state dependencies, and invalid-state reporting |
| 6b — complete | Branch noise updates | Per-branch shape/scale, full-joint ratios, independent draw moments/CDFs, loading-only, current-state dependencies, and invalid-state reporting |
| 6c — complete | Spatial mean update | Dense Gaussian conditional and full-joint ratios in both site-bounds modes; 40,000 unbounded draws, current dependencies, and validation |
| 6d — complete | Spatial covariance update | Standardized IW parameters/full-joint ratios, covariance and precision draw means, scalar-projection CDFs, 1–3 dimensions/noninteger df, and failure reporting |
| 6e — complete | Coefficient-variance updates | Complete joint-GP quadratic and full-joint ratios, library plus field shape/scale, 40,000 draws per branch mode, current/cache dependencies, fixed-jitter consistency, and singularity reporting |
| 7 — complete | Fixed-proposal collapsed random walk | Dense sequential-proposal/acceptance agreement in both branch and coordinate modes; four-chain scalar posterior moments/CDF agree with quadrature; RNG/JIT/vmap, current conditioning, rejection, and validation checks |
| 8 — complete | Complete outer sweep/restart | Exact schedule/new-condition dependency checks in both branch/bounds modes; joint reference moments/cross moments preserved; rejected-site coefficient refresh; bitwise chunk/checkpoint continuation for typed/legacy keys; atomic-failure and incompatible-checkpoint checks |
| 9 — complete | Warmup/frozen production | Standard online moments match independent NumPy covariance; D07 ridge/scale, initial-period and zero-variance fallback, repeated rejections, phase/count boundaries, frozen production, and bitwise warmup checkpoint continuation for typed/legacy keys |
| 10 — complete | Collapsed MALA | Independent dense drift/forward-reverse acceptance in both branch/coordinate modes; standard BlackJAX identity reference; quadrature posterior moments/CDF; current-input Gibbs schedule, rejected-state coefficient refresh, frozen production, and bitwise checkpoint restart |
| 11 — complete | Uncollapsed NUTS and default window adapter | Standard kernel/window-driver agreement (12/60/300 steps), current-condition outer schedule, independent quadrature joint moments, frozen production and bitwise typed/legacy warmup restart |
| 12 — complete | Matched collapsed NUTS | Both NUTS paths pass matched standard-kernel, posterior, current-conditioning, freeze/restart gates; collapsed c_f independence and divergent-state refresh verified |
| 13 — complete | Simplified collapsed MMALA | Independent conditional/Fisher/derivative and normalized asymmetric-proposal references in both bounds/branch modes; quadrature joint posterior invariance, current-input Gibbs order, standard epsilon DA and bitwise freeze/restart |
| 14a — complete | Fixed synthetic-data preprocessing (user-selected first dataset) | R spline equivalence, local OLS on irregular/per-curve depths with near-zero-variance failure, regular candidate-offset grid and slope-error diagnostics, library standardization, 20/60 row and coordinate alignment, QR identities, evaluation-only truth, source hashes and reproducible archive; 123 tests pass |
| 14b — design discussion | Pilot comparisons on Unity | Agree on common target, initial states, resources/budget, convergence and efficiency criteria before submission |
| 15a | Profile-driven optimization | Optimized paths match retained references |
| 15b | Linux HPC execution | Pinned environment and interrupted-run recovery |

Optimize factor reuse, diagonal-output algebra, and unnecessary matrix materialization early. Profile the cost of factoring an $(n-1)k$-dimensional complement at every site before implementing rank updates. Defer approximations, custom derivatives, alternative parameterizations, and multi-device execution until justified. No milestone is complete before its acceptance tests pass.

## 11. Highest risks

1. Confusing library standardization with optional bounds, or imposing artificial support in unbounded mode.
2. Omitting library evidence from the inferred coefficient-variance posterior.
3. Reintroducing a statistical nugget through undocumented numerical repairs or masking structural singularity.
4. Using stale coefficients after partial collapse or stale covariance/gradient caches after variance updates.
5. Confounding calibration/discrepancy identification or proposal/blocking effects with sampler performance.
6. Confusing input sample scaling ($r-1$) with profiled variance estimation ($r$), or freezing profiled variances in the calibration chain.

## 12. Stage 14a data and next task

Stage 14a used the supplied synthetic dataset. The two BKA CSVs contain 20 library loading curves; the two syn_theta_spatial CSVs contain 60 field loading curves. All depth grids have 501 points from 0 to 400 at spacing 0.8. `2026BKA.xlsx` Sheet1 A1:D21 supplies Case IDs 1–20 and physical EA/EM/etr library inputs; `theta_spatial.csv` has 60 corresponding parameter rows for evaluation. Apply the user's field row order: rows 1–20 have y=12, 21–40 y=6, 41–60 y=0, with x=0,6,...,114 in each block. CSV row numbers exclude the header. No unloading curves are present, so this dataset uses the already specified loading-only model (k=5).

Confirmed by the user: apply the note's 4-unit slope matching; add N(0,1²) observation noise with an explicit seed before both matching and projection; proceed with units unspecified and preserve numeric load scale. Seed 1024 was proposed for reproducibility. The same realized noisy field curves must feed matching and projection. The nearby legacy R script does not perform slope matching or offset search: it extracts loading curves and divides depth by its maximum. Its five-site subset and coefficient rescaling are not authorized for this implementation.

Stage 14a alignment decision, most recently simplified at the user's request on 2026-09-28: estimate each field or library slope by intercept-inclusive SciPy local linear regression on **observed** points within its first four depth units or [h0,h0+4], respectively. The matcher accepts a shared depth vector or per-curve depth matrices with irregular spacing, repeated depths, and unsorted rows. It rejects a local slope only when fewer than two points are present or the local depth standard deviation is at most 32 float64 eps times max(1, max absolute depth), making Var(h) numerically zero. Candidate offsets span the library runs' common full-window domain on an evenly spaced NumPy `linspace` grid. `candidate_step=0.8` is the default maximum spacing, independently of observed sample spacing; the actual spacing is at most this value and both domain endpoints are included. Users can set `--candidate-step` for a different preprocessing resolution. At each candidate, average all estimable library slopes and minimize the absolute difference from the mean field slope; discard candidates where any library slope is unestimable. The first candidate wins an exact score tie. The former events/midpoints enumeration is removed. The true raw-point OLS objective is piecewise constant between window-membership changes, so the regular grid is an explicit approximation to its continuous-domain minimum. The slope-error figure shows the evaluated grid points. Only the library load baseline L(h0) is linearly interpolated after alignment.

With JAX typed key seed 1024 and N(0,1²) noise added before matching and QR projection, the observed mean field slope is about -0.012086731926665378. On the default 0.8 candidate grid over [0,396], the selected library offset is h0=0 with absolute mean-slope error about 0.012086731926485059. A previous denser event/midpoint search selected h0≈0.4 with an error smaller by only about 2.4e-13; the chosen grid resolution and near-flat noisy early signal limit physical interpretation of these offsets. The original noiseless curves all share the 0:0.8:400 grid and first exceed |load|=1e-6 at depth 4.8; that threshold was only an audit diagnostic. The current diagnostic figure shows the full regular grid and a zoom around the selected offset.

The prepared artifact in `artifacts/stage14a/synthetic_preprocessing.npz` contains physical and standardized library inputs, five library spline coefficients per run, noisy field curves and the realized noise, site-major projected observations and block-diagonal R, the specified 60 spatial coordinates, normalized/original grids, slope-search diagnostics, and evaluation-only field truth. JSON metadata embedded in the archive records branch orientation, shape/stacking, the exact alignment and QR rules, no unit labels or load rescaling, source SHA-256 hashes, PRNG implementation/seed, and software versions. `artifacts/stage14a/slope_error.png` is the review figure. Source CSV/XLSX files are read-only and unchanged. Reproduce with `UV_CACHE_DIR=/tmp/bayesian-calibration-uv-cache uv run --extra preprocessing python -m bayesiancalibration.data "<supplied synthetic directory>" artifacts/stage14a`.

This task uses the loading-only model with k=5. The source-specific synthetic loader and subsequent common-grid QR fit still validate this particular dataset's shared sorted grid; the generalized matcher alone does not constitute a full real-data preprocessing pipeline. The earlier discussion of a baseline/change-point method pertains to future experimental data with an uncertain origin; it does not alter this source-note preprocessing. Its fit window and post-contact model remain future statistical decisions. Stage 14b sampling/convergence comparisons are next. They require fixed priors, initialization, and other run settings before launching pilots.

### Stage 14b comparison protocol — execution checks complete; production pending

The user requested comparison of MH, MALA, simplified MMALA, NUTS, and collapsed NUTS, followed by selection of 1–2 methods. On 2026-09-28 the user selected **four chains × five methods, six production hours per chain, and unbounded site coordinates**, and explicitly deferred direct Unity submission: prepare the shell scripts only. The user subsequently approved execution-check preparation with 1,000 warmup sweeps, 100 initial covariance sweeps for MH/MALA, initial MALA/MMALA epsilon 0.1, and fixed MMALA epsilon_G=1e-6. No Unity connection or submission has been performed.

Comparison design and approved execution settings:

- Use four independent chains per method (20 jobs), one process per chain. Reuse four dispersed, valid full model initial states across methods by chain index, but give each method/chain an independent explicit PRNG stream. Do not initialize or tune using `theta_spatial.csv` truth.
- Freeze the Stage 14a prepared dataset, library-only standardization and fitted GP length scales once for all methods; keep source-note priors, spatial range 12, support, variance/discrepancy updates, float64 and numerical policy identical. Fit GP length scales once from the library before launch. The user selected unbounded sites; no workbook-header or library-extrema bounds are inferred.
- Compare the current implemented strategies: collapsed sequential site MH/MALA/MMALA versus uncollapsed/ collapsed all-site theta NUTS within the same Gibbs schedule. Current NUTS is not NUTS on the full joint model state. Record this blocking distinction rather than attributing every difference purely to proposal type.
- First perform a short execution check (20 completed warmup sweeps for one chain per method, plus compilation) on the intended compute-node type to measure memory and sweep cost. This is execution/timing evidence, not convergence evidence. Use it to finalize requested cores, memory, walltime and attainable warmup length before the 20-job pilot.
- Approved pilot warmup is 1,000 complete sweeps per chain, subject to the execution-cost check. Retain each method's existing warmup algorithm; MALA/MMALA target acceptance 0.574, both NUTS targets 0.8 with matched diagonal adaptation and tree settings. MH/MALA begin with identity proposal/preconditioner matrices and 100 initial covariance sweeps. MALA/MMALA initial epsilon is 0.1, fixed MMALA epsilon_G is 1e-6, and NUTS initial step size is 1.0, with maximum 10 doublings and divergence threshold 1,000. The exact settings and source-note priors are explicit in `experiments/comparison.json`. No production adaptation. A diagnosed tuning failure permits a documented retuned run; include tuning effort in practical cost comparisons.
- The approved production budget is six hours per chain: 120 chain-hours plus compilation, warmup and I/O; multiply by allocated cores to obtain allocated core-hours. Equal production time gives different retained sweep counts; checkpoints permit extending runs. Use the same CPU/node class and allocated resources, avoid thread oversubscription, and synchronize JAX before timing. Record compilation, warmup, production, I/O and total allocated core-hours separately. Scheduler wait is reported separately from sampler compute time. Exact resource requests depend on Unity's allocation/partition and limits, which are not yet known.
- Save every retained sweep without thinning. Diagnose each method jointly across its four chains using rank-normalized split/folded R-hat, bulk/tail ESS, MCSE, trace/rank plots and relevant numerical diagnostics. Proposed screening targets are R-hat < 1.01, aggregate bulk/tail ESS >= 400 for predeclared quantities, and mean MCSE/posterior SD <= 0.05 where finite. These are screening criteria, not proof of convergence. Require investigation of production NUTS divergences and numerical failures; tree-cap frequency is also an efficiency diagnostic. Do not apply a whole-target energy diagnostic blindly across changing Gibbs conditionals.
- Diagnose all physical theta components (60 × 3), spatial means/covariance scales/correlations, branch noise, coefficient variances, discrepancy, and predeclared spatial contrasts/predictive summaries; retain c_f diagnostics as well. Compare posterior means, intervals and distributions between converged methods using Monte Carlo uncertainty; no method is presumed to be a gold standard. Truth RMSE/coverage is a secondary model check, not the sampler-ranking criterion for this single synthetic dataset.
- Among methods with compatible posteriors and adequate diagnostics, compare low-quantile/minimum and median bulk/tail ESS per allocated core-hour, production ESS/time and end-to-end time to required MCSE, memory, and chain-to-chain stability. When pooling four chains, divide pooled ESS by summed compute cost; report parallel time-to-result separately. Do not discard a method solely because an equal-time pilot has insufficient ESS; distinguish slow exploration from failure and extend promising cases before selecting 1–2 methods. Retain all chains, including poorly mixing ones, in reports.

Remaining deployment inputs: Unity partition, optional account/CPU-class constraint, cores, memory and total scheduler walltimes. These are deliberately required environment values for the command-printing script. The production scheduler walltime must exceed six hours by compilation, warmup and I/O costs measured on the compute-node execution check. The scripts do not connect, install packages, or submit jobs. Resource selection and the actual execution checks/pilot remain pending; no sampler ranking or convergence is claimed.

#### Prepared implementation and artifacts

- `experiments/comparison.json`: explicit shared model, approved tuning, 21,600-second production budget, four chains, seed 20260928, and chunk size 10. Library-only optimization uses three log-length starts (-1,-1,-1), (0,0,0), (1,1,1), gtol=1e-6, ftol=1e-10, maxiter=1000, no search bounds and zero jitter. These numerical optimizer settings do not add priors or constraints.
- `comparison.py prepare`: reads only the five model input arrays from the Stage 14a archive; evaluation truth is excluded from the frozen experiment. Fits the library once and stores four full initial states, configuration, fit attempts, preprocessing metadata/source hashes, and implementation hashes in a pickle-free archive. Initialization uses independent prior draws of mu_theta and eta|mu_theta with C_theta ⊗ I, Sigma_theta=I, delta=m_delta_0, sigma_y2=1, sigma_c2=library column mean squares, and exact conditional draws of c_f. All initial states must validate without retries or regularization. These are starting values only; all variables follow the existing Gibbs updates. Each chain index shares exactly the same start across methods; method/chain streams are independent explicit fold-ins, separate from the initialization stream.
- `artifacts/stage14b/prepared.npz`: prepared locally from the approved Stage 14a artifact. All three fits converged to essentially the same objective; selected standardized lambda_c=(3.35581539, 6.82735774, 6.89863374), objective=-409.5710751294. All four initial states passed the existing full-boundary checks. This is input preparation, not a real-size MCMC run.
- `mcmc.ChunkRunner`: reuses JAX sweep/adaptation functions between chunks; its transitions, validation, warmup rules and frozen production are the existing drivers. The sweep is compiled before timed chunks without consuming PRNG keys. Adaptation compilation belongs to warmup. Static settings cannot change within a runner. No changing model cache is stored.
- `comparison.py run`: one process per method/chain; complete warmup and production chunks saved separately, no thinning. NPZ chunks retain every model variable and sampler diagnostic, full-joint log density, iteration and phase. A single-writer file lock protects each chain. Write the chunk first, then atomically commit the existing checkpoint with a chunk hash list, configuration and cumulative clocks. On restart verify all committed files and resume the exact boundary. An uncommitted chunk is ignored/replayed; missing or damaged committed data is rejected. SIGINT/SIGTERM/SIGUSR1 requests a stop after the current chunk. Hard termination can lose that chunk, whose wasted cost must still be included from scheduler accounting.
- Production budget counts synchronized production-driver time, including checks/stacking but excluding sweep compilation, warmup and disk writes. Stop after the last started complete chunk, with at most one chunk of overshoot; reduce to one sweep near the boundary. Saved production time is cumulative across restarts. Session JSON records sweep compilation/setup, I/O, driver wall time, environment/host/backend and status; driver wall time begins after input loading/initial-chain construction. **Slurm accounting supplies total end-to-end allocated cost and peak memory**, including input loading, Python startup, killed/uncommitted work, and resubmissions. Compilation is recorded per process, so restart costs remain visible. The checkpoint's production clock is the committed useful-work clock, not a replacement for allocation cost.
- `scripts/unity_comparison.sh`: CPU Slurm worker, one process per task, explicit float64, nested BLAS thread caps and `srun --cpu-bind=cores`. Task IDs 0–3=MH, 4–7=MALA, 8–11=MMALA, 12–15=NUTS, 16–19=collapsed NUTS; chain index is task ID modulo four. The 20-sweep check uses IDs 0,4,8,12,16 and preserves the full 1,000-sweep adaptation schedule; it is not a shortened production run or convergence check. Check and production output directories are separate.
- `scripts/print_unity_commands.sh`: prints escaped `sbatch` commands only, with configurable resource requests and concurrency. No `sbatch` execution occurs. OSU Unity's Slurm use is documented in [ASC Unity Cluster](https://math.osu.edu/sites/default/files/2021-04/Unity%20Cluster%20SLURM%2BLMOD.pdf); no other institution's Unity partitions/accounts are assumed. JAX timing follows its [benchmarking guidance](https://docs.jax.dev/en/latest/201/profiling.html).

#### Hot-path and warmup policy update (2026-09-28)

The user requested consistent removal of per-sweep defensive validation across all five methods, followed by the explicit revised zero-scatter policy below. No numerical model or proposal formula was changed.

- Full model-shape/dtype/PRNG/static tuning validation remains in preparation, initialization, explicit validators, and checkpoint serialization/reconstruction. Public drivers require initialized/restored chains; manually altered dataclasses must be explicitly validated again before running. Drivers keep only cheap type/phase/counter checks. There is no full-chain validation inside a sweep loop. The reusable runner retains a cheap static-settings identity check so the wrong sampler/target configuration cannot silently reuse a compiled closure.
- A completed chunk performs one GP geometry audit (including the existing unjittered covariance criterion). The drivers no longer recompute the complete joint density, gradient, every MMALA site metric, or adaptation schedule just to validate each completed sweep. The full-joint density already returned by the numerical sweep remains a saved diagnostic. Checkpoint I/O continues to validate its external boundary; this cost is separately timed.
- JAX `checkify` carries device-side finite/positive-state predicates and diagnostic checks to the host, which raises before recording an invalid sweep. Current MALA/NUTS gradients are checked where the algorithm already computes them, including after changing Gibbs conditioning. MMALA checks the actual forward/reverse metric-derived mean/diffusion rather than building an extra all-site host metric scan. Warmup checks new covariance factors/adaptation numbers and epsilon or mass representability without repairing them. Raw pure kernels remain usable with JIT/vmap; their `debug_check` predicates are activated by the checked drivers (or explicit `checkify.checkify`).
- Removed target-level blanket conversion of every NaN/Inf to -inf. Numerical failures now remain visible. MH/MALA/MMALA target evaluations distinguish NaN/+inf (error) from -inf (ordinary rejection). Current states must always be finite. Standard BlackJAX MALA candidate-energy/gradient rejection and NUTS trajectory divergence handling remain intrinsic algorithm behavior; invalid current gradients cannot be hidden by those mechanisms. Divergent NUTS trajectories can retain a valid state and continue with the divergence recorded. No library NUTS tree or accept/reject rule was replaced.
- Retained explicit D07 regularization, the approved fixed MMALA epsilon_G, configured fixed GP jitter policy (benchmark value zero), BlackJAX window/mass regularization, and algebraic covariance symmetrization. Removing these would change the declared algorithm/numerical policy. There are no added retries, clipping, tolerance-based scatter floors, or replacement of nonzero invalid covariance estimates with old tuning.
- **Latest zero-scatter decision:** an eligible MH/MALA update with exactly zero empirical covariance holds the previous valid proposal during adaptive warmup and increments `zero_covariance_count` for that site. At the final warmup boundary, any exactly zero site fails with `WarmupTuningError`; production is not entered. This also applies to a schedule with no adaptive covariance interval. Warmup completion is a tuning gate, not a claim of convergence.
- `RandomWalkAdaptationState` now persists per-site `acceptance_count`, `movement_count`, and `zero_covariance_count`. Acceptance is the MH decision; movement is an actual change in eta, including the initial-state-to-first-sweep transition. They are intentionally distinct. The covariance estimator still excludes the initial position as originally specified; thus a single initial movement followed by identical retained warmup states can have positive movement count but zero empirical covariance. That warmup still fails the covariance gate. No tolerance is used to classify movement or exact zero scatter.
- Checkpoint schema **4**, covariance-adaptation protocol **v2**, records these counters and checks their integer ranges at serialization/restart. Old checkpoints are rejected rather than migrated. All sampler output records include per-site movement; the comparison archive retains it per sweep and aggregates phase-specific movement/acceptance counts (MH decision counts apply to MH/MALA/MMALA; NUTS retains its native acceptance statistic). Warmup tuning counters are also in the comparison checkpoint/session record. A failed final warmup writes `warmup_failure.json` with site indices and counts; it cannot create production draws. The last committed valid partial checkpoint stays intact.
- The frozen `artifacts/stage14b/prepared.npz` was regenerated with current implementation hashes; all 34 input/initial-state arrays were verified bitwise identical to the previous archive. All methods must use this same new archive and implementation. Existing raw/preprocessed data, experiment model/tuning settings, `.gitignore`, and unrelated user edits are preserved. No Unity benchmark or new sampler ranking is authorized by this refactor.

#### Usage after transferring the repository and frozen artifact

Create the Linux environment once, outside timed jobs, using the repository lock (`uv sync --locked --python 3.13`). The shell worker uses `.venv/bin/python`; `COMPARISON_PYTHON` can select another equivalent environment. Do not copy the macOS virtual environment. Transfer the same source tree, `uv.lock`, and prepared archive to every job; do not refit separately by method. The archive refuses changed implementation hashes. If code/configuration changes before launch, create a new prepared archive and experiment output directory.

To prepare a new archive explicitly (the default artifact above already exists):

```bash
uv run --locked python -m bayesiancalibration.comparison prepare \
  --preprocessing artifacts/stage14a/synthetic_preprocessing.npz \
  --config experiments/comparison.json --output artifacts/stage14b/prepared-new.npz
```

After choosing measured resources, export `PARTITION`, `CPUS`, `MEMORY`, `CHECK_WALLTIME`, and `PRODUCTION_WALLTIME`; optionally `ACCOUNT`, `CONSTRAINT`, and `MAX_PARALLEL`. Then print commands for review:

```bash
bash scripts/print_unity_commands.sh artifacts/stage14b/prepared.npz artifacts/comparison
```

No connection or submission is performed by this command. Review the execution-check results before using the printed production command. Identical output paths automatically resume committed work. For allocation cost/memory after future execution, retain `sacct -j JOB_ID --units=K --format=JobID,State,ElapsedRaw,AllocCPUS,TotalCPU,MaxRSS` including job-step records, and sum allocation time over attempts; do not double-count job and step allocation rows.

**Next task:** investigate the collapsed-NUTS conditional-GP/invalid-state failures before its first mass update, persistent uncollapsed-NUTS tree caps after mass learning, and MALA’s unmoved site. Further diagnostics, tuning changes, and production require a subsequent instruction; no additional runs are pending. Kronecker choice 1 is implemented as `M^{-1} = Gamma_site ⊗ Gamma_param`, both estimated from eta warmup moments. MH initial variance remains 1e-6. Production remains deferred.

References for the proposed diagnostics: [Stan diagnostic guidance](https://mc-stan.org/learn-stan/diagnostics-warnings.html) and [Vehtari et al., rank normalization, folding, and localization](https://arxiv.org/abs/1903.08008).

## 13. Supporting references

- [Van Dyk and Park](https://www.ma.imperial.ac.uk/~dvandyk/Research/08-jasa-pcg.pdf) and [Van Dyk and Jiao](https://arxiv.org/abs/1309.3217): partial-collapse ordering and MH composition.
- [R spline basis documentation](https://stat.ethz.ch/R-manual/R-devel/library/splines/html/bs.html): approved five-column convention.
- [JAX JIT](https://docs.jax.dev/en/latest/jit-compilation.html), [PRNG](https://docs.jax.dev/en/latest/random-numbers.html), and [benchmarking](https://docs.jax.dev/en/latest/benchmarking.html): implementation checks.
- [JAX multivariate normal](https://docs.jax.dev/en/latest/_autosummary/jax.random.multivariate_normal.html): standard Cholesky Gaussian draws for coefficient refresh and discrepancy updates.
- [BlackJAX NUTS](https://blackjax-devs.github.io/blackjax/autoapi/blackjax/mcmc/nuts/index.html) and [window adaptation](https://blackjax-devs.github.io/blackjax/autoapi/blackjax/adaptation/window_adaptation/index.html): standard transitions, default tuning, and cached state.

## 14. Status and validation record

| Event | Record |
| --- | --- |
| Initial review | Seven source notes reviewed; architecture and validation roadmap established. |
| D01–D10 consolidation | Standardized priors, branch noise, explicit targets, and decision register incorporated. |
| R01–R04 consolidation | Applied R spline convention, removed statistical nugget, made coefficient variances sampled state with their full conditional, and replaced mandatory unit-box bounds with library standardization plus optional support. Updated caches, schedule, tests, and roadmap. |
| U01–U02 consolidation | Recorded library sample standardization with $r-1$ scales and explicit overrides; derived the selected profile-likelihood objective with $q_j/r$ nuisance variances and log-length-scale optimization. Updated preparation order, cache dependencies, tests, and roadmap. No numbered review decisions remain open. |
| Stage 1 implementation | Added Cholesky-based normalized Gaussian log densities, collapsed projected moments/density, and conditional coefficient moments/density in `src/bayesiancalibration/linalg.py`. Added the deterministic $n=3,k=2$ fixture and reference tests in `tests/test_model.py`. No GP, MCMC, adaptation, or real-data code was added. |
| Stage 1 validation | `UV_CACHE_DIR=/tmp/bayesian-calibration-uv-cache uv run python -m unittest discover -s tests -v` passed all five tests. Tests cover independent SciPy joint/marginal densities, NumPy conditional moments, the Gaussian factorization identity, unequal branch variances, loading-only mode, float64, and JIT execution. `git diff --check` passed. |
| Implementation status | Stages 1–6 complete; Stages 7–15b not started. |
| Document validation | Decision coverage, stale-assumption scan, and unrelated-file preservation checked. |
| 2026-09-24 source-notation rule | Read the four revised notes; added the durable rule to `AGENTS.md`, the notation crosswalk above, and explicit reconciliation items. Checked names against source meanings and Stage 1 shapes. Documentation-only change; Python APIs and prior test results unchanged, tests not rerun. |
| 2026-09-24 main-note and sampler decisions | Reread the four main notes; classified all other notes as archived references only. Updated site-only bounds, always unbounded `mu_theta`, Gaussian mean update, NUTS, and default BlackJAX window adaptation across the model, schedule, validation requirements, and roadmap. Verified defaults against installed BlackJAX 1.6.2 and official documentation; retained the joint-restriction convention. Stale-assumption scan and document whitespace check passed. Documentation-only update; Python and source notes unchanged, tests not rerun. |
| Stage 1 notation refactor | Renamed model-specific arguments and locals in `linalg.py` and updated the existing test fixtures/callers. Documented source meanings, shapes, stacking, and full-covariance aliases; generic helpers and numerical operations are unchanged. `UV_CACHE_DIR=/tmp/bayesian-calibration-uv-cache uv run python -m unittest discover -s tests -v` passed all five tests, including SciPy/NumPy reference identities, loading-only behavior, float64, and JIT. Keyword-call tests use the new API. Old-name scan and whitespace checks passed; Stage 2a remains next. |
| 2026-09-25 standard-function rule | Added the rule to prefer JAX/SciPy/NumPy functions, restrict custom helpers to documented needs or clear benefits, and briefly explain each custom helper. Documentation-only change; Stage 2a remains next. Checked the edited text and whitespace; tests not rerun because executable code was unchanged. |
| 2026-09-25 Stage 1 standard-function refactor | Replaced the two-step Cholesky solve helper with JAX `cho_solve` and the hand-written normalized Gaussian log-density helper with JAX `multivariate_normal.logpdf`. Model-specific projected moments and density functions retain their source-note notation and behavior. All five existing unit tests passed, including SciPy/NumPy reference values and JIT; JIT-compiled float64 gradients of the collapsed density with respect to its mean and covariance were finite. Stage 2a remains next. |
| Stage 2a–2b implementation | Added frozen float64 library standardization and validated overrides in `state.py`; added physical/standardized site maps, finite-box and identity computational coordinates, stable log-Jacobian, and standardized spatial-prior configuration in `transforms.py` and `state.py`. No GP, MCMC, or real-data preprocessing was added. |
| Stage 2a–2b validation | `UV_CACHE_DIR=/tmp/bayesian-calibration-uv-cache uv run python -m unittest discover -s tests -v` passed 11 tests. New tests compare sample mean/$r-1$ scales and affine maps with NumPy, reject invalid samples/overrides/bounds, check physical prior-center conversion, both coordinate modes, box Jacobian by finite differences, JIT, and unclipped tails. Whitespace and unrelated-edit checks passed. Stage 3a remains next. |
| Stage 3a–3b implementation | Added the unit-amplitude ARD kernel, fixed library factor/cache, exact field conditional moments, and normalized fixed-library likelihood in `gp.py`. Added zero-mean library-only profile objective over log length scales and multistart SciPy L-BFGS-B fitting with JAX gradients and recorded diagnostics. Output variances remain dynamic inputs to the GP rather than frozen profile estimates. |
| Stage 3a–3b validation | `UV_CACHE_DIR=/tmp/bayesian-calibration-uv-cache uv run python -m unittest discover -s tests -v` passed 17 tests. Six new GP tests compare dense joint/conditional/library Gaussian calculations with SciPy/NumPy, verify JIT and variance dependence, check jitter sensitivity and structural-singularity rejection, match analytic $q_j/r$ to numerical variance optimization and the full library likelihood, compare log-length gradients with finite differences, exercise loading-only columns, validate fit diagnostics, and reject zero coefficient columns. Whitespace and unrelated-edit checks passed. Stage 4 remains next. |
| Stage 4 implementation | Added fixed `CalibrationTarget` specification in `targets.py` with one joint spatial field, theta-only and full collapsed/uncollapsed densities, current coefficient variances, branch-specific observation noise, standardized prior factors, and finite-box Jacobian. Inverse-Gamma and inverse-Wishart densities are explicit JAX formulas because the installed JAX stats module lacks them. No sampler or coefficient refresh was added. |
| Stage 4 validation | `UV_CACHE_DIR=/tmp/bayesian-calibration-uv-cache uv run python -m unittest discover -s tests -v` passed 25 tests. Eight new target tests compare full densities with SciPy in bounded and unbounded modes, check the joint/marginal/conditional identity and one-site collapsed ratio against independent conditional Gaussians, compare JIT gradients with finite differences, verify current-variance dependence, unbounded spatial mean, loading-only mode, configured priors/range, and invalid variance handling. Python compilation and whitespace checks passed. Stage 5 remains next. |
| Stage 5 implementation | Added pure JAX exact coefficient correction draw and a checked host refresh in `gibbs.py`. Both use explicit keys and standard Gaussian draws/Cholesky solves. Made the existing observation-array builder public in `targets.py` to share branch stacking with Gibbs draws. Current theta, discrepancy, noise, and coefficient variances are recomputed; structural GP singularity is reported without repair. No other Gibbs updates or outer sweep were added. |
| Stage 5 validation | `UV_CACHE_DIR=/tmp/bayesian-calibration-uv-cache uv run python -m unittest discover -s tests -v` passed 31 tests. Six new tests compare analytic moments against the source-note precision form, check 40,000-draw empirical means/covariances within six Monte Carlo standard errors in two-branch and loading-only modes with varying non-diagonal site designs, verify explicit-key reproducibility and float64 JIT/vmap, compare checked refresh with independently assembled GP/observation moments in both coordinate modes, check all changing-state dependencies, and reject invalid variances and singular field/library coincidences even with jitter. Python compilation and whitespace checks passed. Stage 6a remains next. |
| Stage 6a implementation | Added pure discrepancy conditional moments and Gaussian draw plus checked host `update_discrepancy` in `gibbs.py`. Standard JAX Cholesky/triangular solves and `random.multivariate_normal` implement the specified Gaussian conditional at fixed current field coefficients. Documented prior/conditional notation and target prior fields. No branch-noise updates or outer sweep were added. |
| Stage 6a validation | On 2026-09-26, `UV_CACHE_DIR=/tmp/bayesian-calibration-uv-cache uv run python -m unittest discover -s tests -v` passed 37 tests. Six new tests compare moments with independent observation-space Gaussian conditioning under default and configured correlated/nonzero priors, check a one-site closed form, match full-joint discrepancy ratios in bounded and unbounded site modes, and verify 40,000-draw means/covariances within six Monte Carlo standard errors for branch sizes (5,5) and (5,) with varying non-diagonal site designs and unequal noise. Tests also cover float64 JIT/vmap, explicit-key reproducibility, current coefficient/noise dependencies, invalid input rejection, and nonfinite draw reporting. Python compilation, whitespace, and unrelated-edit checks passed. Stage 6b remains next. |
| Stage 6b implementation and validation | Added active-branch inverse-Gamma parameter/draw kernels and checked `update_branch_noise`. Only projected residuals contribute; the target `alpha_y_0`/`beta_y_0` fields are prior parameters. JAX has no inverse-Gamma sampler, so a documented log-Gamma transformation provides explicit-key float64 JIT/vmap sampling without floors. The four Stage 6b tests in `test_gibbs_updates.py` passed: independent NumPy shape/scale and full-joint ratios in both site-bounds/branch modes, 40,000 draws per branch mode with SciPy mean/CDF checks and branch independence, near-unit/small-shape tail checks over a wide scale range, key/current-state dependencies, and failure reporting. Stage 6c is next in the authorized sequence. |
| Stage 6c implementation and validation | Added joint-field Gaussian spatial-mean moments/draw and checked `update_spatial_mean`. Standard Cholesky solves use the full C_theta; mu_theta is unbounded in both site-bounds modes. A small shared host coordinate checker keeps the three spatial/GP update boundaries consistent. Four targeted SpatialMeanTest tests passed: dense joint-Gaussian moments, full-joint ratios with means outside the site box in both modes, 40,000-draw Gaussian moments with bounded sites and an outside-box prior center, current eta/covariance dependencies, explicit-key reproducibility, and invalid covariance/site rejection. Stage 6d is next in the authorized sequence. |
| Stage 6d implementation and validation | Added standardized spatial-covariance IW parameters/draw and checked `update_spatial_covariance`. JAX lacks an IW sampler; a documented Bartlett construction using standard normal/chi-square draws and triangular solves preserves explicit JAX keys and JIT/vmap without matrix inverses, jitter, or redraws. Four targeted SpatialCovarianceTest tests passed: dense scale/df and full-joint ratios in both bounds modes with outside-box means, 40,000 draws checked against SciPy covariance means and Wishart precision means plus scalar-projection CDFs, one/three-dimensional and noninteger-df laws, keys/current-state dependencies, and invalid/nonfinite-state reporting. Stage 6e is next in the authorized sequence. |
| Stage 6e implementation and validation | Added coefficient-variance IG parameters/draw and checked `update_coefficient_variances`, including fixed library evidence and current field evidence. Five targeted CoefficientVarianceTest tests passed: complete dense joint-GP quadratics and full-joint ratios in both bounds/branch modes, 40,000 draws per branch mode with SciPy means/CDFs and Gamma precision means, current sites/coefficients and per-column dependencies, rebuilt library caches, fixed positive-jitter joint consistency, explicit keys/JIT, and invalid/structurally singular/nonfinite state reporting. All requested Stage 6 substeps are complete. |
| Stage 6b–6e final regression validation | On 2026-09-26, `UV_CACHE_DIR=/tmp/bayesian-calibration-uv-cache uv run python -m unittest discover -s tests -v` passed all 54 tests, including 17 new tests in `tests/test_gibbs_updates.py`. Substeps were implemented and their acceptance tests passed in order 6b, 6c, 6d, 6e before the full regression run. Float64/JIT/vmap and both branch/site-bounds modes are covered. Python compilation, line-length/whitespace checks, and unrelated-edit checks passed. Production host wrappers and pure numerical kernels remain separate; outer sweeps and samplers are later milestones. Stage 7 remains next. |
| Coefficient-variance notation refactor | Renamed the GP variance vector to `sigma_c2` and its prior parameters to `alpha_c_0`, `beta_c_0` across APIs, target fields, call sites, derived test variables, and this plan. Generic distribution symbols and unrelated R plotting variables retain their meanings. On 2026-09-26, `UV_CACHE_DIR=/tmp/bayesian-calibration-uv-cache uv run python -m unittest discover -s tests -v` passed all 54 tests. Repository searches including hidden and ignored project sources found no stale coefficient-variance references. Python compilation and line-length checks passed; an AST comparison after reversing the identifier renames and normalizing string whitespace confirmed unchanged calculations and control flow in all eight edited Python files. Stage 7 remains next. |
| Branch-specific coefficient priors | User decision: `alpha_c_0` and `beta_c_0` are `(B,)` arrays, analogous to branch-noise priors. Updated target construction/validation, normalized joint priors, Gibbs parameter/draw APIs, and branch mapping; defaults remain 1.01/0.01 per branch. Explicit scalar inputs are rejected. On 2026-09-26, `UV_CACHE_DIR=/tmp/bayesian-calibration-uv-cache uv run python -m unittest discover -s tests -v` passed all 56 tests. Validation covers distinct branch priors, unequal branch sizes, loading-only mode, dense joint-GP conditionals and full-joint ratios, JIT/vmap sampling, default float64 arrays, and rejection of scalar, wrong-shaped, nonpositive, or nonfinite priors. Python compilation, line-length/whitespace checks, and the stale scalar-API search passed. Stage 7 remains next. |
| Prior/conditional naming consistency | Target prior fields, constructor keywords, validation messages, Gibbs wrappers, and test references now use `m_delta_0`, `V_delta_0`, `alpha_y_0`, `beta_y_0`, `alpha_c_0`, and `beta_c_0`. Names without `_0` denote full-conditional quantities only. Removed obsolete field-alias descriptions and aligned prior equations. On 2026-09-26, `UV_CACHE_DIR=/tmp/bayesian-calibration-uv-cache uv run python -m unittest discover -s tests -v` passed all 56 tests. A repository scan including hidden and ignored project sources found unsuffixed names only for full-conditional quantities and no obsolete prior field references. Python compilation and line-length/whitespace checks passed. Stage 7 remains next. |
| Stage 7 implementation | Added `samplers/metropolis.py` with fixed `(n,d,d)` eta-space proposal covariances, sequential site updates, separate diagnostics, and fresh density initialization after conditioning changes. Standard JAX Gaussian draws and BlackJAX MH implement proposals and acceptance; custom code only supplies the model target and site schedule. Host validation checks current inputs, proposal SPD, and initial/final unjittered GP geometry without repair. The five Stage 7 acceptance tests passed; complete outer-sweep/restart integration remains Stage 8. |
| Stage 7 validation | On 2026-09-26, `UV_CACHE_DIR=/tmp/bayesian-calibration-uv-cache uv run python -m unittest discover -s tests -v` passed all 61 tests, including five new tests in `tests/test_random_walk.py`. Independent NumPy/SciPy joint-GP and spatial densities match site-by-site probabilities, decisions, and final positions for sequential updates under both branch/coordinate modes with non-diagonal site proposals. Scalar calibration posterior means, second moments, and CDFs agree with independent SciPy quadrature within six batch-mean Monte Carlo standard errors, both pooled and per chain: four chains per coordinate mode, 2,000 discarded fixed-proposal sweeps and 24,000 retained sweeps each. Rejected states are retained. Tests cover explicit-key replay, JIT/vmap/float64, all changing conditioning inputs, extreme-proposal rejection without clipping, invalid shapes/variances/covariances, and structural-singularity detection even under configured library jitter. Compilation, line-length/whitespace, and unrelated-edit checks passed. Stage 8 remains next. |
| Stage 8 implementation | Added completed `CalibrationState`, fixed random-walk chain state, pure full Gibbs sweep, checked chunk driver, and atomic pickle-free checkpoint save/load. The outer schedule follows the current Sampling note, and coefficients refresh immediately after collapsed theta using the updated sites/variances. The next unused PRNG key, iteration, phase, tuning, fixed specification, package/code/source hashes, configuration, and backend/dtype provenance are persisted. The seven Stage 8 acceptance tests passed; Stage 9 is next. |
| Stage 8 validation | On 2026-09-26, `UV_CACHE_DIR=/tmp/bayesian-calibration-uv-cache uv run python -m unittest discover -s tests -v` passed all 68 tests, including seven new tests in `tests/test_mcmc.py`. The complete pure sweep agrees with separate checked conditional updates and their current inputs in both bounds/branch modes. Rejected theta states remain recorded while coefficients refresh; invalid states/counts and injected numerical failures cannot return a partial chain. Joint stationarity was checked independently in both bounds modes against 150,000 NumPy/SciPy importance-reference draws (effective sample size above 30%): 40,000 posterior-start chains preserve all seven variable means, theta/coefficient second moments, and theta–coefficient, theta–mean, coefficient–discrepancy cross moments through four complete sweeps within six combined Monte Carlo standard errors. Globally restricted joint support is applied before weighting, and fixed-library evidence is included. Five uninterrupted sweeps match two sweeps plus checkpoint/load plus three sweeps bitwise for typed and legacy keys, including every sample, diagnostic, final key, and iteration. Corrupt payloads, incompatible target/schema/protocol/code/package/JAX configuration, invalid phase/iteration, source-hash mismatches, and failed atomic replacement are covered. Python compilation and line-length/whitespace checks passed; unrelated user files remain untouched. Stage 9 remains next. |
| Stage 9 implementation decisions | Warmup and initial fixed-period lengths are explicit positive inputs with `num_initial <= num_warmup`. Identity is the note-prescribed default initial proposal; a declared validated SPD override is allowed. Standard dense BlackJAX Welford estimators count all completed warmup eta states, including repeated rejections and initial-period states, excluding the initialization position; covariance uses sample denominator count minus one. After the initial period, apply exactly D07 and the 2.38-squared/d random-walk scale when the estimate is usable; otherwise retain the previous SPD proposal. Equal warmup/initial lengths keep initial tuning throughout. Adapt after completed sweeps only; the last warmup state freezes tuning/statistics without an extra draw. Rename the chain container to `RandomWalkChain` to reflect both phases, and extend checkpoints to schema 2 with separate adaptation state; schema 1 is not migrated automatically. |
| Stage 9 validation | On 2026-09-27, the eight tests in `tests/test_adaptation.py` and the full `UV_CACHE_DIR=/tmp/bayesian-calibration-uv-cache uv run python -m unittest discover -s tests -v` suite passed (76 tests). Online means/M2 and D07 proposals agree with independent NumPy sample covariances, including repeated states and large position offsets. Zero variance retains previous SPD tuning without NaN factorization; positive-trace rank-deficient estimates receive only the specified ridge. Initial periods, identity defaults, short/fixed-only warmup, invalid schedules/counters/phases, rejected states with coefficient refresh, exact freeze of tuning/statistics in production, and chunk-boundary enforcement are covered. Interrupted warmup plus checkpoint/load and production matches uninterrupted execution bitwise for typed/unbounded dual-branch and legacy/bounded loading-only cases, including every sample, diagnostic, final model/key, and iteration; frozen-boundary save/load and incompatible adaptation metadata are covered. JIT roundoff asymmetry found during validation is removed by symmetric triangle mirroring of covariance statistics/proposals. Existing Stage 8 stationarity and checkpoint regression tests remain passing. Python compilation and line-length/whitespace checks passed. Stage 10 is next. |
| Stage 10 implementation decisions | Use standard BlackJAX MALA in local affine Cholesky coordinates, with `step_size=epsilon^2/2`, to obtain the source-note dense covariance/drift and exact forward/reverse correction. Reinitialize density/gradient at every site from the latest other sites and Gibbs inputs; local zero coordinates preserve rejected states exactly. Require declared positive scalar epsilon, fixed in both phases. MALA empirical `V_prop` is D07 preconditioner P (multiplier 1.0), with epsilon-squared applied in the transition; the random-walk 2.38-squared/d scale remains specific to random walks. Reuse the Stage 9 estimator/schedule and share only complete-sweep orchestration. Add separate MALA chain factories/drivers and sampler-tagged schema-2 checkpoints with epsilon in the checksummed array payload. No new dependencies or statistical model changes are introduced. |
| Stage 10 validation | On 2026-09-27, all 11 tests in `tests/test_mala.py` passed, followed by the full `UV_CACHE_DIR=/tmp/bayesian-calibration-uv-cache uv run python -m unittest discover -s tests -v` suite (87 tests). Five-point finite differences of independent NumPy/SciPy joint-GP/spatial densities validate the dense drift and both Gaussian proposal densities, with sequential acceptance/position agreement under both branch and coordinate modes and non-diagonal preconditioners. Identity preconditioning agrees with standard BlackJAX MALA. Scalar calibration posterior means, second moments, and CDF agree with independent quadrature within six batch-mean Monte Carlo standard errors, pooled and per chain, using four chains per coordinate mode with 2,000 discarded and 24,000 retained sweeps each. Tests cover explicit-key/JIT/vmap replay, all current conditioning inputs, invalid scales/states/gradients, structural GP singularity under both jitter settings, exact repeated rejections without clipping, current-input full Gibbs schedule and immediate coefficient refresh, MALA-specific D07 scaling, adaptation boundaries, and frozen epsilon/preconditioners/statistics in production. Typed/unbounded dual-branch and legacy/bounded loading-only interrupted warmup plus checkpoint/load and production match uninterrupted execution bitwise, including samples, diagnostics, tuning, final model/key, and iteration. Epsilon payload corruption and sampler/tuning mismatches are rejected. Existing random-walk warmup, joint stationarity, and checkpoint regression tests remain passing. Python compilation, line-length/whitespace, and `git diff --check` passed; unrelated user files remain untouched. Explicit epsilon selection and real-data convergence assessment remain caller responsibilities. Stage 11 is next. |
| MALA epsilon adaptation decisions | User requested standard BlackJAX dual averaging during warmup. The initial opt-in choice was superseded by the subsequent user request: `target_accept` now defaults to 0.574, and only explicit None disables epsilon adaptation. Use one common scalar epsilon, updated after each complete sweep from mean site acceptance probabilities, including the initial covariance period. Standard defaults t0=10, gamma=0.05, kappa=0.75 apply directly to epsilon; MALA integrator step_size remains epsilon-squared/2. Pass epsilon dynamically into the JIT sweep. Keep D07 covariance adaptation unchanged and simultaneous; no additional windows, resets, clipping, or draws. Freeze the standard final averaged epsilon and both adaptation states at the sampling boundary. Persist DA configuration and all standard state scalars in the checksummed checkpoint payload, with a separate protocol tag. The requested extension is complete. |
| MALA epsilon adaptation validation | On 2026-09-27, all six tests in `tests/test_mala_adaptation.py` passed, then the full `UV_CACHE_DIR=/tmp/bayesian-calibration-uv-cache uv run python -m unittest discover -s tests -v` suite passed (93 tests). Per-sweep model states, keys, diagnostics, epsilon, and DA statistics agree with the standard BlackJAX utility and explicit dynamic-epsilon full sweeps. Nonbinary site probabilities affect feedback, and the updated kernel differs from a captured-initial-epsilon reference. Short warmup uses the library final average with no extra draw; epsilon, covariance moments, and DA statistics remain exactly frozen during production. Tests reject invalid targets, scalar dtypes/configuration, centers/counters, phase/epsilon mismatches, corrupt or unconfigured checkpoint payloads, and unrepresentable adapted scales without clipping or a partial returned chain. Coupled covariance/epsilon adaptation matches uninterrupted warmup and production bitwise across initial, partial-warmup, and frozen-boundary save/load for typed/unbounded dual-branch and legacy/bounded loading-only cases, including every sample/diagnostic, tuning, DA state, final model/key, and iteration. All existing fixed-epsilon MALA, random-walk, posterior/quadrature, and joint-stationarity tests remain passing. Compilation, changed-file line-length/whitespace checks, and `git diff --check` passed; unrelated user edits remain untouched. Standard DA defaults and an explicit target are computational tuning choices; changing-conditional warmup does not establish real-data convergence. Stage 11 remains next. |
| MALA default target decision | User requested `target_accept=0.574` by default. Omission now enables standard dual averaging; explicit None retains fixed-epsilon warmup. Existing fixed-epsilon tests declare None, while the standard-DA full-sweep reference omits the argument and verifies the default 0.574 behavior. Production freeze and checkpoint behavior are unchanged. This default change is complete. |
| MALA default target validation | On 2026-09-27, `UV_CACHE_DIR=/tmp/bayesian-calibration-uv-cache uv run python -m unittest discover -s tests -p 'test_mala*.py' -v` passed all 17 relevant tests. The existing per-sweep standard-DA reference now omits target_accept and verifies target 0.574, dynamic epsilon feedback, and final averaging. Fixed-epsilon cases explicitly request None. Posterior/quadrature, covariance adaptation, production freeze, checkpoint/restart, invalid configuration, and failure checks remain passing. Compilation, changed-file formatting, stale-documentation search, and `git diff --check` passed. |
| Stage 11 implementation | Added the all-site site-major `nuts_sweep` and `NUTSChain` factories, warmup/production drivers, scalar trajectory diagnostics and per-transition step/mass tuning. The shared outer schedule keeps current c_f fixed through the uncollapsed tree and refreshes it immediately afterward, using split-8-v1. Every tree calls standard BlackJAX NUTS initialization with current conditioning, without an endpoint MH step. The exact BlackJAX 1.6.2 staged Welford-diagonal engine embeds standard identity/1.0/0.8/1000 defaults, 75/25/50 windows, short-run handling and regularization. Welford ignores gradient, so Gibbs adaptation passes zero for that unused argument after coefficient refresh. Final averaged step/mass and all adaptation statistics freeze before production. Schema 3 stores standard DA/Welford window statistics, full resolved schedule/defaults, tree/energy limits, and hashes/version/configuration; no changing target cache is stored. The low-level staged engine is version-sensitive and validated in the installed 1.6.2 environment; other library versions require their own equivalence validation. |
| Stage 11 validation | On 2026-09-27, all six `test_nuts.py` tests passed within the full 99-test run; the only regression failure was an old schema-2 assertion, updated for schema 3, after which all eight `test_adaptation.py` tests passed. Standard diagonal-mass kernels agree in both coordinate and branch modes; 12/60/300-step fixed Gaussian warmup matches the standard library window driver including slow-window resets, regularized inverse masses and final averages. Equivalent comparisons use matching dynamic JIT inputs, avoiding constant-folding differences. Explicit current-input Gibbs references validate the ordering/key protocol and immediate c_f refresh. Independent NumPy/SciPy quadrature and posterior starts preserve theta/c_f means, second/cross moments and CDF through three transitions for 12,000 independent chains per bounds mode within six Monte Carlo standard errors; this is invariance evidence, not mixing evidence. Initial/mid/finished warmup checkpoints and frozen production match uninterrupted runs bitwise for typed/unbounded dual-branch and legacy/bounded loading-only cases. Invalid tuning, counters, phase/chunk boundaries and payload corruption are rejected. Compilation, line-length and whitespace checks passed. Stage 12 starts next. |
| Stage 12 implementation | `collapsed=True` selects the integrated coefficient target in the same `nuts_sweep`, `NUTSChain` factories and drivers. All-site blocking, standard window engine/defaults, tree limits, PRNG protocol, full Gibbs schedule, immediate exact coefficient refresh, recording and validation remain matched. Checkpoint sampler tags distinguish `collapsed_nuts` from `uncollapsed_nuts`; density/gradient initialization follows the selected current conditional every sweep. This isolates the target-collapsing choice for later real-data comparisons. |
| Stage 12 validation | On 2026-09-27, `UV_CACHE_DIR=/tmp/bayesian-calibration-uv-cache uv run python -m unittest discover -s tests -p '*nuts.py' -v` passed all 14 tests. Both paths match standard BlackJAX block kernels for loading-only/dual-branch and bounded/unbounded fixtures, explicit current-input outer updates and coefficient refresh, scalar theta/c_f quadrature posterior invariance, invalid boundaries and payload checks, and bitwise initial/mid/final warmup checkpoint continuation with frozen production. Resolved 1000-step default adapter states/settings are identical before warmup; both use the same independently validated window engine. The collapsed theta kernel is exactly independent of old c_f, responds to each observation/spatial/variance conditioning input, and divergent repeated theta states still receive new coefficients. Compilation/whitespace checks passed. Stage 13 starts next; neither NUTS path establishes convergence on real data. |
| Stage 13 implementation | Added `samplers/mmala.py` with site-major conditional observation moments, `jax.jacfwd` mean/covariance derivatives, the source-note Fisher metric, transformed spatial conditional precision, bounded Jacobian curvature, and required positive `epsilon_G I`. Nonunit standardized widths use sigmoid p; unbounded coordinates use identity D and zero curvature. Standard Cholesky/`cho_solve`, JAX Gaussian generation, SciPy-compatible Gaussian logpdf and BlackJAX asymmetric RMH implement the simplified drift/diffusion and both proposal directions, including normalization determinants. Candidates recompute their own metric using the latest other sites; no metric-volume target term or Christoffel drift is added. `MMALAChain`/warmup/production drivers share the complete outer schedule and immediate c_f refresh. Only epsilon uses standard DA with mean site probabilities and default target 0.574; final average and statistics freeze, while the position-dependent metric continues to be evaluated. Ridge is a required per-run computational input and remains fixed. Schema 3 stores MMALA epsilon/ridge and separate epsilon-only schedule/standard DA state with sampler-specific protocol tags and checksums; no empirical covariance adaptation or changing metric cache is stored. |
| Stage 13 validation | On 2026-09-27, the five `test_mmala.py` tests and four `test_mmala_adaptation.py` tests passed, followed by the full `UV_CACHE_DIR=/tmp/bayesian-calibration-uv-cache uv run python -m unittest discover -s tests -v` suite: **116 tests passed** in 225 seconds. Dense independent NumPy Gaussian conditionals and five-point derivatives validate conditional means/covariances and the complete Fisher metric for both bounds/branch modes, with nonzero covariance derivative effects. A separate scalar analytic derivative reference validates identity coordinates and nonunit 1.55 standardized width. Independent forward/reverse drift/covariance/logpdf calculations match sequential Gaussian proposals, normalized determinant corrections, probabilities and decisions with changing metrics, acceptance and rejection. Explicit-key/JIT/vmap replay, current eta/noise/spatial covariance/coefficient variances, discrepancy/mean gradient effects and observed-complement metric dependence are covered. Invalid ridge/epsilon, structural repeated-site singularity and extreme proposal rejection are tested without repair. Three transitions of 12,000 independent posterior-start chains per bounds mode preserve theta/c_f means, second/cross moments and CDF against independent quadrature within six Monte Carlo standard errors. Full checked-update references verify all current conditioning, split-8-v1, exact coefficient refresh and finite final joint density, including rejected theta. Dynamic-epsilon sweeps match standard DA feedback/final average; single-step and chunk boundaries are correct; production freezes epsilon/ridge/DA while recomputing geometry. Typed/unbounded dual-branch and legacy/bounded loading-only initial/mid/final warmup plus checkpoint/load/production agree bitwise in samples, diagnostics, model/key, tuning/statistics and iteration. Fixed-tuning restart, corrupt ridge, incompatible sampler/schedule/statistics and invalid phase/counts are rejected. All existing target, Gibbs, joint-stationarity, RW/MALA warmup and both NUTS regressions pass. Compilation, changed-file line-length/whitespace checks and `git diff --check` pass; unrelated user edits are preserved. No real-data convergence, cross-platform bitwise equivalence, or untested BlackJAX-version compatibility is claimed. At that point Stage 14a was next. |
| Stage 14a implementation | Added `data.py` with standard SciPy cubic B-spline design matrices matching R’s five-column basis; loading-only library QR coefficients, field QR projection, library-only frozen sample standardization, all 20/60 source rows and stated coordinates, seed-1024 JAX float64 field noise, source/version/PRNG/configuration metadata, pickle-free NPZ and two-panel slope-error PNG. Local OLS uses `scipy.stats.linregress` on observed depth/load pairs without uniform/strict grid assumptions. A 2026-09-28 simplification removed event/midpoint enumeration and uses an explicitly configured regular `np.linspace` candidate grid, default maximum step 0.8. `openpyxl` and `matplotlib` are optional `preprocessing` dependencies; no sampler or statistical target changed. |
| Stage 14a validation | Seven R 4.6.1 `splines::bs` values including endpoints and knots agree with SciPy to 1e-14; the first supplied library curve’s five `qr.solve` coefficients agree with Python to 7.3e-12. The prior local-OLS implementation passed all 123 tests in 241.435 seconds on 2026-09-27. On 2026-09-28, the simplified regular-grid matcher passed seven focused tests covering irregular/repeated/unsorted per-curve depths, nonmultiple spacing, independent SciPy regression for every candidate, grid regularity/configured resolution and invalid step, numerically degenerate local depth failure, full source shapes/standardization, QR identities, evaluation-only truth, and pickle-free output. The regenerated default artifact has F_s (20,5), y_tilde (300,), R (300,300), 496 evenly spaced slope candidates, seed-1024 noise empirical SD 1.0014, and selected h0=0 with error 0.01209. Full/local scatter panels were visually inspected. The source-specific loader still expects the supplied 20/60-row, 501-column shared-grid dataset; other real-data input and QR handling remain future work. Stage 14b remains unrun. |
| Stage 14b execution preparation (2026-09-28) | Implemented the five-method/four-chain CLI, frozen preparation artifact, reusable chunk kernels, synchronized cumulative production budget, atomic draw/checkpoint commits, restart verification, independent streams, and print-only Unity shell commands. Approved settings are unbounded sites, 1,000 warmup sweeps, MH/MALA initial covariance period 100, initial MALA/MMALA epsilon 0.1, MMALA epsilon_G=1e-6, and 21,600 production seconds per chain. Three library fits converged consistently and four shared full initial states validated. No Unity connection or submission, compute-node check, or real-size MCMC production was performed. |
| Stage 14b preparation validation | `UV_CACHE_DIR=/tmp/bayesian-calibration-uv-cache uv run python -m unittest discover -s tests -v` passed all 129 tests in 266.978 seconds. Six new tests cover the complete task/independent-key mapping, identical starting states, bitwise chunk/public-driver agreement for all five methods across warmup and production, committed-time restart without budget reset, corrupt draw rejection, replay after a failed checkpoint write, evaluation-truth exclusion, and shell syntax/print-only behavior (including spaces in paths and a fake sbatch that must never execute). Default CLI task-19 dry run, Python syntax, shell executable modes, and whitespace checks passed. Resource choices, compute-node feasibility checks, posterior diagnostics and sampler selection remain pending. |
| Pre-benchmark hot-path simplification | Removed full-chain validators, duplicated current gradients/metrics and per-sweep GP spectrum checks from all five driver loops. Full static validation remains at initialization and external checkpoint boundaries; dynamic device predicates use standard JAX checkify and GP geometry is audited once per completed chunk. Targets preserve NaN/Inf instead of blanket conversion to -inf. Intrinsic proposal rejection, NUTS divergences, MMALA metric construction, and declared regularization remain. No retry, clipping, or nonzero invalid-covariance replacement was added. |
| Revised zero-scatter warmup decision | The user's latest instruction supersedes the intermediate strict-zero failure request: exactly zero empirical covariance temporarily retains prior valid tuning and increments a site-level issue counter during adaptive MH/MALA warmup. Terminal zero scatter raises WarmupTuningError before production. Acceptance, actual movement, and zero-hold counts persist with schema 4/protocol v2 and appear in comparison output; final failures write warmup_failure.json. The estimator still excludes the initial position, while movement counts include the first transition. |
| Hot-path regression validation | The full 137-test run completed in 311.721 seconds: 135 passed and two old short-warmup success expectations failed under the new terminal-zero policy. Updated those tests to assert terminal failure or use a two-state nondegenerate warmup. Reran all eight adaptation tests (21.943 seconds) and all six MALA adaptation tests (26.696 seconds), all passing; the latter uses a 1e-14 relative tolerance for compiled-versus-eager standard DA rounding while still checking exact frozen production/restart. Thus all 137 cases are validated across the full run plus affected-suite reruns. Eight new hot-path cases passed: no static validators during sampling; one geometry audit per chunk for all methods; nonfinite Gibbs draws and current gradients cannot be hidden; actual invalid MMALA metrics fail; NUTS divergence remains diagnostic; temporary zero scatter can recover with exact persisted counters; nonzero invalid covariance cannot fall back; and final warmup failure saves site counts without production output. Syntax, whitespace, source scans and prepared-artifact load checks passed. Refreshed common preparation has all 34 numerical input/initial-state arrays bitwise unchanged. No Unity execution or benchmark timing claims. |

### Unity execution check (2026-09-28; complete)

The user authorized environment setup, artifact verification, and only the five-method 20-sweep check, superseding the earlier submission restriction for these checks. Production remains deferred. The workspace is already on `unity-login1.asc.ohio-state.edu`. Installed project-local `venv/tools/uv` 0.12.19 and managed CPython under `venv/python`; `UV_CACHE_DIR=/tmp/bayesian-calibration-uv-cache UV_PYTHON_INSTALL_DIR=/home/kim.9859/calibration-random-effects/venv/python venv/tools/uv sync --locked --python 3.13` created `.venv`. An offline locked sync verified the installation. Python 3.13.15, JAX/JAXlib 0.11.2, BlackJAX 1.6.2, NumPy 2.5.3, and SciPy 1.18.1 are installed. The lockfile is unchanged (SHA-256 `32298fee549734ee82af44d4435208c265ee4edbc9fdffd747408a6363298156`).

The user transferred `artifacts/unity-check/prepared.npz`, SHA-256 `cd25b8796f97233e7886ee6d13a66845c18f74695d9dd308107b2e400b642bd3`. No refit or sampler/model changes were made. Compute-node preflight job 11190895 verifies CPU float64 JIT, archive implementation hashes and configuration, all four initial states, and initialization of chain 0 for each method. Resource choice: account `stat-users`, partition `stat-cascade`, 4 cores and 16 GiB per process, CPU binding and single-threaded nested BLAS. Evidence is saved under `results/unity-check-20260928/`. The intended check array is exactly `0,4,8,12,16`, with 20 warmup sweeps, the unchanged 1,000-sweep adaptation schedule, and zero production draws.

Preflight passed on `u118` (exit 0): 40.75 s process wall time and 616,496 KiB peak RSS from GNU time. Submitted check-only Slurm array `11190896` with IDs `0,4,8,12,16`, `--constraint=cascade`, 4 CPUs, 16 GiB and 30-minute wall limit per task. No production jobs were submitted. Runtime/memory and numerical outcomes remain pending.

Initial array `11190896` failed before Python startup (all five tasks: exit 2, 1 s allocated each): `Missing Python environment: /var/spool/slurmd/.venv/bin/python`. Slurm executes a spool copy, so resolving the repository from `BASH_SOURCE` was incorrect. Updated only `scripts/unity_comparison.sh` to use the `--chdir` working directory, with optional `COMPARISON_REPO` override. Shell syntax, a spool-copy/fake-srun reproduction verifying the Python path and exact `--check-sweeps 20` arguments, and `git diff --check` passed. Python implementation hashes and frozen inputs are unchanged. Resubmitted exactly the same check array as `11190903`; no sampling occurred in the first attempt.

Final array `11190903` completed successfully on `u118` for all five tasks (exit 0); each saved exactly iterations 1–20 in warmup, two committed ten-sweep chunks, and zero production time/draws. Each task waited 1 s before allocation. The preflight waited 7 s. Final Slurm accounting and per-session clocks are saved in `results/unity-check-20260928/accounting.psv` and `summary.json`; raw stdout/stderr, checkpoints, and draw chunks are retained there.

| Method | Slurm elapsed (s) | Sweep compilation (s) | 20-sweep driver (s) | Checkpoint/I/O boundary (s) | Peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| mh | 16 | 3.515 | 2.999 | 0.170 | 569.2 |
| mala | 38 | 4.092 | 20.463 | 0.297 | 659.5 |
| mmala | 127 | 6.512 | 72.510 | 14.738 | 905.5 |
| nuts | 101 | 4.380 | 84.430 | 0.327 | 688.6 |
| collapsed_nuts | 132 | 4.444 | 114.847 | 0.334 | 686.3 |

Elapsed is end-to-end allocated wall time, including Python startup, input loading, initialization, compilation and I/O. Sweep compilation is explicitly timed before chunks; the 20-sweep driver includes adaptation compilation and chunk audits. The checkpoint/I/O clock also includes full checkpoint-boundary validation; MMALA's 14.738 s is not pure disk time. RSS is the sampling step's Slurm-reported MaxRSS. Successful checks used 1,656 allocated core-seconds (0.460 core-hours); the failed launcher attempt used 20 core-seconds, and preflight used 164 core-seconds, for 1,840 core-seconds (0.511 core-hours) total, without double-counting step rows. These short warmup timings do not establish production throughput or six-hour memory requirements.

Diagnostics: MH accepted 0/1,200 site proposals, with all 60 sites unmoved and mean acceptance probability 1.98e-36. MALA accepted 549/1,200 (45.75%), with 4 sites unmoved; MMALA accepted 693/1,200 (57.75%), with all sites moving. Uncollapsed NUTS had 7/20 divergent transitions and 13/20 maximum-doubling hits; collapsed NUTS had 6/20 divergences and 14/20 maximum-doubling hits. These are the first 20 steps of the unchanged 1,000-step adaptation schedule, not production diagnostics. MH/MALA's 100-step initial covariance period has not ended; zero-scatter terminal handling was not exercised. No retuning, retries of sampler transitions, model changes, convergence claims, or sampler ranking were made.

Validation: compute-node Python/package imports, CPU float64 JIT Cholesky reference, prepared implementation hashes and exact configuration, all four initial states and all five method factories passed. After execution, verified every saved draw-chunk SHA-256 against the committed checkpoint, contiguous iterations 1–20, warmup phase, check limit 20, finite model states/full joint log densities, zero production clock, all five successful Slurm exits, and empty sampling stderr. Shell syntax and spool-copy launcher verification passed. `uv.lock`, prepared archive, and all Python source remain unchanged. Environment setup emitted a harmless cross-filesystem hardlink-to-copy warning; the initial empty bootstrap-directory error was corrected before installation. The only failed compute jobs were the documented launcher-path attempt. The user queue is empty; no full production jobs were submitted.

### Reduced-scale 100-sweep warmup checks (2026-09-28; execution finished; two failed)

The user requested several reductions of MH `initial_proposal_variance`, smaller initial NUTS/ collapsed NUTS steps, and 100 warmup sweeps. Operational tuning grid: MH variances `1e-2`, `1e-4`, `1e-6` (proposal standard deviations 0.1, 0.01, 0.001; original variance 1); both NUTS initial steps `0.1`, `0.01` (original 1). These seven chain-0 runs start afresh from the frozen initial states, with the same method-specific keys as the previous checks and across scale variants. This pairs the tuning comparisons; variants are not independent replicate chains. No MALA/MMALA or production jobs are included.

Retain the full 1,000-step adaptation schedule, MH's initial covariance period of 100, NUTS target 0.8 and maximum 10 doublings, all model choices, and the original library fit. Stop with `--check-sweeps 100`; this is not a complete 100-step warmup schedule and does not freeze tuning for production. The MH check reaches its first empirical-covariance update at sweep 100, but does not evaluate subsequent sweeps with that covariance.

Created five separately identified frozen archives under `results/unity-tuning-100-20260928/`, each changing only the named configuration value. All 34 numerical input/initial-state arrays were checked bitwise identical to the original archive; implementation hashes match, no library refit or reinitialization occurred, and the original archive/default config remain unchanged. `manifest.json`, per-variant configs and `prepare_variants.py` record provenance. Added `COMPARISON_CHECK_SWEEPS` to the existing shell worker (default remains 20). Shell syntax, spool-copy/fake-srun checks of default 20 and override 100, rejection of invalid counts, and whitespace checks passed. Planned resources remain 4 CPUs, 16 GiB, `stat-cascade`/`cascade`, account `stat-users`, 30 minutes per task, CPU float64 and single-threaded nested BLAS.

Submitted check-only jobs: `11191280` (mh-variance-1e-2), `11191281` (mh-variance-1e-4), `11191282` (mh-variance-1e-6), `11191283` (nuts-step-0p1), `11191284` (nuts-step-0p01). Each MH array contains task 0; each NUTS array contains tasks 12 and 16. No production submission occurred.

Interim MH results: all three checks completed 100 sweeps and exited 0. Variance `1e-2`: 24/6,000 accepted site proposals (0.4%), 56 sites never moved, 56 zero-scatter holds at the first covariance update; `1e-4`: 574/6,000 (9.567%), 37 unmoved sites/holds; `1e-6`: 2,484/6,000 (41.4%), no unmoved sites or zero-scatter holds. Slurm elapsed times were 34, 29 and 30 s respectively. The zero-scatter retention is the existing approved temporary policy, not an execution failure at sweep 100; the terminal check remains at sweep 1,000. NUTS checks remain in progress.

Interim failure: collapsed NUTS with initial step 0.1 (`11191283_16`) raised `FloatingPointError: Gibbs sweep produced invalid model state` while executing the third chunk (sweeps 21–30); only sweeps 1–20 are committed. No invalid chunk was saved, no automatic sampler retry or numerical/model modification was made. The exact failing sweep/quantity is not identified by the existing aggregate predicate. Other checks continue.

Final results: all seven submitted jobs terminated. Three MH and two uncollapsed-NUTS checks completed 100 sweeps with exit 0; both collapsed-NUTS checks failed with exit 1. No jobs were resubmitted after these numerical failures. The interrupted final accounting query was successfully retried after the user's instruction to continue; it did not interrupt or restart sampling. Final Slurm accounting is `results/unity-tuning-100-20260928/accounting.psv`; `summary.json` and `summarize.py` retain the diagnostics, validation, resource measurements, and complete error text.

| Method | Changed initial tuning | Committed sweeps | Slurm elapsed (s) | Committed sweep driver (s) | Peak RSS (MiB) | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| mh | 0.01 | 100 | 34 | 13.155 | 575.2 | COMPLETED |
| mh | 0.0001 | 100 | 29 | 13.753 | 575.3 | COMPLETED |
| mh | 1e-06 | 100 | 30 | 14.032 | 559.7 | COMPLETED |
| nuts | 0.1 | 100 | 506 | 486.748 | 693.3 | COMPLETED |
| collapsed_nuts | 0.1 | 20 | 181 | 117.687 | 713.6 | FAILED |
| nuts | 0.01 | 100 | 548 | 520.498 | 762.6 | COMPLETED |
| collapsed_nuts | 0.01 | 70 | 521 | 424.907 | 861.4 | FAILED |

Tuning is proposal **variance** for MH and initial **step size** for NUTS. Slurm elapsed includes startup, compilation, I/O and failed uncommitted work; the committed sweep clock excludes the failed chunk. All tasks used 4 CPUs/16 GiB with nested BLAS caps. Jobs ran on Cascade Lake nodes `u118` and `u117`; queue wait was 3–4 seconds. Total allocation cost across all seven attempts, including failures, was 7,396 core-seconds (2.0544 core-hours); no step rows were double-counted. These timings are early warmup measurements, not production throughput.

Final diagnostics:

- MH variance 1e-2: acceptance 24/6,000 (0.4%), 56 of 60 sites never moved, 56 zero-scatter holds at sweep 100. Variance 1e-4: 574/6,000 (9.567%), 37 unmoved sites/holds. Variance 1e-6: 2,484/6,000 (41.4%), all 60 sites moved, no zero-scatter holds. The 1e-6 setting removes the all-site startup sticking in this check but does not establish mixing: its standardized endpoint RMS displacement is only 0.00555. No next-step use of the newly adapted covariance was tested; no default tuning was changed.
- Uncollapsed NUTS initial step 0.1: 11/100 divergences, 90/100 tree-cap hits; last 20 sweeps had 1 divergence and 19 cap hits. Initial step 0.01: 7/100 divergences, 94/100 cap hits; last 20 had 0 divergences and 20 cap hits. Last-used step sizes were 1.8932e-7 and 2.7505e-7 respectively; these are adaptive warmup values, not frozen production tuning. Smaller initial steps did not resolve persistent tree-depth saturation.
- Collapsed NUTS initial step 0.1 failed in sweeps 21–30 with `FloatingPointError: Gibbs sweep produced invalid model state`; 20 sweeps committed, including 5 divergences and 15 cap hits. The existing predicate detects nonfinite model entries or nonpositive observation/coefficient variances but does not identify the particular quantity or exact failing sweep.
- Collapsed NUTS initial step 0.01 failed in the field-site covariance audit after the 71–80 chunk with `ValueError: Unjittered C_f_given_s is singular or numerically singular`; only 70 sweeps committed, including 12 divergences and 58 cap hits. The rejected chunk is not counted as validated draws. No jitter increase, target change, or failed-transition retry was introduced.

Validation completed: variant archive hashes, every committed draw-chunk hash against its checkpoint, contiguous iterations through each committed boundary, warmup phase and check limit 100, finite stored model states/full joint log densities, and zero production clocks. All five successful jobs have empty stderr and matching successful Slurm statuses; both failures and full stderr are retained. The saved-run audits passed; this does not make either failed sampler run successful. The original prepared archive SHA-256 remains `cd25b8796f97233e7886ee6d13a66845c18f74695d9dd308107b2e400b642bd3`; all numerical Python source and the lockfile are unchanged. Shell syntax and whitespace checks passed. User queue is empty. Full production remains unsubmitted.


### 2026-09-28 — Conditional whitened NUTS and MH startup variance (whitening subsequently deferred)

User decision: MH initial proposal covariance is now `1e-6 I_d`, overriding the identity startup in the source note and earlier plan. Apply conditional affine whitening to both collapsed and uncollapsed NUTS. The new whitened section in `Sampling theta.md` was read; its uncollapsed formula is extended to the collapsed target as explicitly requested. Implementation and the stated reference/integration validation are complete. Full-data convergence/performance assessment remains pending.

Use site-major `eta = mu_theta + L_C @ z @ L_Sigma.T`, equivalent to the specified Kronecker map. Triangular solves recover z. Recompute the map after current Gibbs mean/covariance updates, hold it fixed through the complete NUTS tree, and return eta to the existing coefficient refresh/model state. Compose the existing target with this map; the omitted affine log-Jacobian is constant during each conditional update. This is not a joint reparameterization of the hyperparameter updates. In unbounded mode the spatial prior is standard normal in z; with finite bounds the literal affine eta map remains valid but is not prior whitening of the logit-normalized target. Preserve its existing logit Jacobian and support without changing the model.

Adapt diagonal inverse mass from the actual selected z and acceptance statistic, with the existing BlackJAX window schedule, regularization and step adaptation. Freeze step and z-space mass after warmup; the conditional coordinate map still changes when Gibbs hyperparameters change. Record selected z in NUTS diagnostics and identify the coordinate convention in checkpoint metadata. Existing code-hash enforcement and the coordinate tag prevent reuse of eta-space tuning/checkpoints. Keep MALA's identity initial preconditioner unchanged. The comparison config declares `mala_initial_proposal_variance=1.0` separately; legacy configs without this key retain their shared-value behavior. No new jitter, dependency, or production submission.

Acceptance checks passed: dense Kronecker/triangular-solve agreement; standard-normal prior gradient/Hessian; direct BlackJAX kernel agreement under both targets and bound modes; current-conditional z adaptation; outer Gibbs ordering; exact restart and frozen tuning; posterior quadrature invariance; MH default and unchanged MALA behavior.

Validation on 2026-09-28 (local locked `.venv`, float64):
- `.venv/bin/python -m unittest discover -s tests -p 'test*nuts.py'`: 18 tests passed (74.447 s), including inherited collapsed-target checks.
- `.venv/bin/python -m unittest discover -s tests -p 'test_adaptation.py'`: 8 tests passed (21.909 s).
- `PYTHONPATH=tests .venv/bin/python -m unittest test_comparison test_hotpath test_mala_adaptation test_mcmc`: 28 tests passed (87.848 s), including runner/checkpoint integration and MALA regression coverage.
- `git diff --check`: passed. Total: 54 focused tests, no failures. The full repository suite was not rerun.

No new real-data/Unity run has been executed. The previous prepared archives/checkpoints are preserved and remain incompatible with changed implementation hashes; a fresh, explicitly recorded preparation is needed for the next comparison. The reference checks establish implementation correctness, not resolution of the earlier 60-site GP singularities, divergence rates, tree caps, or mixing. Next validation is a full-data warmup comparison, including MH beyond its first covariance update; production remains deferred.


### 2026-09-28 — Whitening deferred; selectable eta-space NUTS mass requested

Latest user decision supersedes the preceding whitening task: defer whitening, retain original eta coordinates (including the existing bounded logit target), and allow diagonal, Kronecker, or full dense mass for both NUTS targets. The user explicitly requests clarification before executing ambiguous changes. Only read-only implementation inspection and this planning update have been performed in this turn; the existing whitening code has not yet been removed or replaced.

Unresolved decision: the Kronecker structure and estimator are not specified. Proposed default interpretation is `M^{-1} = A_site ⊗ B_param` with SPD factors `(n,n)` and `(d,d)` learned from centered eta warmup samples, rather than fixing the site factor to the model's `C_theta`. A transparent separable moment estimator from partial traces of the empirical covariance, with factor-scale normalization and declared factor regularization, is a candidate; it is not the general matrix-normal maximum-likelihood estimator. An alternative is a fixed spatial factor `C_theta` with only a parameter factor estimated; bounded eta need not inherit the model's spatial covariance, so these choices impose different tuning restrictions. Obtain the user's choice before implementation.

Proposed shared protocol: diagonal remains the default; identity initialization; existing window schedule/step-size adaptation; BlackJAX's standard covariance estimator/regularization for full dense; freeze mass and step after warmup in all modes. Empirical covariance estimates tune BlackJAX's `inverse_mass_matrix` (`M^{-1}`), not the momentum covariance `M`. No extra bounded transformation or Jacobian is introduced. Retain MH initial variance 1e-6 and MALA settings. Pending acceptance tests include both target/bounds modes, covariance structure and SPD checks, BlackJAX diagonal/dense equivalence, Kronecker dense-reference identities, adaptation freeze, and checkpoint/restart compatibility.


### 2026-09-28 — Approved selectable eta-space masses

The user approved option 1 and requested notation `M^{-1} = Gamma_site ⊗ Gamma_param` (mathematically $\Gamma_{\mathrm{site}}\otimes\Gamma_{\mathrm{param}}$). Both SPD factors are estimated from warmup eta samples; neither is the model's spatial covariance. Whitening helpers, z-space transitions, and z-space adaptation are removed. The existing bounded/unbounded eta target and its Jacobian are unchanged for both collapsed and uncollapsed NUTS.

Public chain factories accept `mass_structure="diagonal" | "kronecker" | "dense"`; comparison JSON uses `nuts_mass_structure` with default `diagonal`. Diagonal and dense use the installed BlackJAX Welford recipes, standard schedule, identity initialization, and dual averaging. All mass tuning and step sizes freeze after warmup. Metadata records the mass structure and `eta-v1` coordinates; changed code hashes and protocol metadata reject older incompatible checkpoints.

Exact Kronecker estimator: for N completed positions in a slow window, use centered sample covariance S (divisor N-1), indexed `S[i,q,j,r]` by site-major stacking. Form `A[i,j] = sum_q S[i,q,j,q]/d` and `B[q,r] = sum_i S[i,q,i,r]/n`. Let `w=N/(N+5)`, `ridge=5e-3/(N+5)`, `A_reg=w*A+ridge*I_n`, `B_reg=w*B+ridge*I_d`. Set `Gamma_site=n*A_reg/trace(A_reg)` and `Gamma_param=B_reg`, then `M^{-1}=Gamma_site ⊗ Gamma_param`. This fixes `trace(Gamma_site)=n`, assigns scale to Gamma_param, and keeps both factors SPD even for rank-deficient/zero empirical scatter. This is an explicitly regularized moment approximation, not a matrix-normal MLE or an exact posterior covariance. With regularization omitted it recovers every exactly separable SPD covariance. The implementation equivalently takes partial traces after the standard dense BlackJAX regularization, whose ridge commutes with these normalized partial traces.

The initial reference implementation reused dense Welford statistics and materialized the Kronecker matrix for the standard BlackJAX kernel; direct factor moments now supersede the dense statistics as documented below. This enforces statistical separability but does not reduce dense storage or factorization cost; no specialized Kronecker momentum/integration kernel is claimed. Validate matrix symmetry/SPD (negative off-diagonal entries are valid), factor structure, warmup state dimensions, and restart identity. No model jitter or extra target term is introduced. MH and MALA settings remain as previously approved.

Validation completed (local locked `.venv`, float64):
- `PYTHONPATH=tests .venv/bin/python -m unittest test_nuts_mass`: initial five tests passed in 49.543 s. These cover independent partial-trace and factor regularization calculations; separable covariance recovery with unequal site/parameter dimensions and negative off-diagonal entries; zero-scatter SPD; dense and single-site Kronecker equivalence to BlackJAX's dense window driver; both targets/bound modes against direct dense NUTS; and exact restart/frozen tuning for both added structures.
- Added comparison dispatch/configuration test passed separately in 3.616 s. All three structures reach both NUTS factories and invalid names are rejected. The test module now contains six tests.
- `PYTHONPATH=tests .venv/bin/python -m unittest test_nuts test_collapsed_nuts test_comparison test_hotpath test_adaptation test_mala_adaptation`: 43 tests passed in 148.693 s, including restored eta-space diagonal reference behavior, posterior quadrature invariance, MH/MALA settings and experiment/checkpoint integration.
- Source compilation and `git diff --check` passed. Total: 49 focused tests, no failures. The full repository suite was not rerun.

No full-data/HPC sampling has been executed in this task. Real-data divergence, tree caps, covariance singularities, and mixing improvements are still unmeasured. At this validation checkpoint Kronecker used dense moments and a dense selected mass; the subsequent direct-moment optimization is recorded below. The resolved formulas, normalization, regularization, and notation are recorded above; there is no outstanding model decision for this implementation.


### 2026-09-28 — Direct online Kronecker factor moments

User requested updating Gamma_site and Gamma_param without first updating a dense covariance. Replace the Kronecker dense Welford scatter by its two sufficient partial traces while preserving the approved moment estimator, regularization, normalization, and window schedule. For the Nth position, let D be the (n,d) eta matrix minus the previous running mean. Update the mean by D/N and update site scatter by `(N-1)/N * D @ D.T`, parameter scatter by `(N-1)/N * D.T @ D`. These are exactly the partial traces of Welford's full centered scatter in exact arithmetic. Form Gamma_site/Gamma_param from these scatters at slow-window boundaries using the previously documented divisors and regularization; the mass stays fixed within each window and throughout production.

New `KroneckerWelfordState` stores mean `(nd,)`, site scatter `(n,n)`, parameter scatter `(d,d)`, and sample count. Accumulator storage decreases from O(n²d²+nd) to O(n²+d²+nd), and scatter update arithmetic is O(n²d+nd²). The resulting selected inverse mass is still materialized for BlackJAX, so whole-chain storage and NUTS factorization costs remain dense. This is an exact sufficient-statistics change, not a new covariance estimator or an alternating matrix-normal fit. The checkpoint protocol is updated to preserve/reload the two scatter arrays and reject incompatible older payloads.

Validation completed (float64, local locked environment):
- `PYTHONPATH=tests .venv/bin/python -m unittest test_nuts_mass`: six existing structure/kernel/freeze/restart tests passed in 49.939 s with direct factor statistics.
- The added `test_online_factor_scatter_matches_centered_batch_at_every_step` passed in 0.692 s: JIT updates match centered batch scatter after every sample, including repeated states and nonzero means; no full scatter array exists; the final metric matches the old regularized dense-partial-trace reference and both scatters reset at the window boundary.
- `PYTHONPATH=tests .venv/bin/python -m unittest test_nuts test_collapsed_nuts test_comparison test_mcmc`: 28 regression tests passed in 102.840 s.
- Total: 35 tests passed, no failures. Source compilation and `git diff --check` passed. No full-suite rerun or full-data/HPC sampling was performed. The only remaining dense allocation in this adaptation representation is the selected inverse mass supplied to BlackJAX; dense NUTS kernel costs and real-data mixing performance remain outside this change.

### 21-way 200-sweep execution checks (2026-09-28; finished: 12 completed, 9 failed)

The user requests MH, MALA, MMALA and both NUTS targets with each diagonal/Kronecker/dense mass: nine chain-0 runs, 200 sweeps each, MH initial variance 1e-6. Production remains deferred. Two execution settings are being clarified: common initial NUTS step (current default 1 versus previously tested 0.1/0.01), and whether 200 means the prefix of the existing 1,000-step schedule or a complete 200-step warmup. No dependent sampling jobs will be submitted until these choices are resolved. MALA retains its separate identity preconditioner; MALA/MMALA epsilons remain 0.1.

The locked environment passes offline sync. The old archive's hashes differ for adaptation.py, comparison.py, mcmc.py, run.py, and samplers/nuts.py, as expected after the user's mass-structure changes. Model/GP code and the comparison build_target/preparation calculations are unchanged. A fresh artifact will explicitly import the frozen library fit and all 34 numerical data/initial-state arrays with bitwise checks and source provenance, validate the target and initial states under current code, and carry current implementation hashes. Old artifacts/checkpoints will remain untouched; no old checkpoint or tuning will resume. Evidence belongs under `results/unity-mass-200-20260928/`.

Resolved by the user: compare all three initial NUTS steps (1.0, 0.1, 0.01) for each of the three mass structures and both NUTS targets, and retain the 1,000-step schedule while stopping after 200 sweeps. Total is 21 chain-0 checks (MH/MALA/MMALA once each plus 18 NUTS combinations). Model, initial states, and per-method keys remain shared across tuning variants. No production.

Compute-node validation passed: job `11193150` ran all seven `test_nuts_mass` cases successfully (124.895 s test time; 131.75 s process time). Preparation job `11193153` revalidated imported frozen inputs under the current implementation, all four states and 21 method/structure/step factories (33.97 s process time). All 34 arrays in each of nine new archives remain bitwise identical to the old archive; source hashes and migration provenance are recorded. Submitted arrays `11193154`–`11193162`, one per mass/step combination: diagonal step 1 includes tasks 0/4/8/12/16; all other arrays include 12/16 only. Explicit check limit 200, 1,000-step schedule, 4 CPUs/16 GiB, 45-minute limit, Cascade Lake partition/account as before. No production.

Interim results: MH completed 200 sweeps, with accepted-site proportions 41.4% for sweeps 1–100 and 19.333% for 101–200; every site moved in each half. Dense collapsed NUTS at initial step 0.1 failed during sweeps 21–30 with the aggregate invalid-model-state check; 20 sweeps are committed. This predates the first mass update, so it cannot establish an effect of the learned dense mass. Other tasks continue; failures are retained without numerical repair or resubmission.

Interim NUTS results: all collapsed-NUTS runs with initial steps 1.0 and 0.1 failed before mass learning: step 1.0 retained 40 sweeps and failed the conditional-GP covariance audit, step 0.1 retained 20 sweeps and failed the invalid-Gibbs-state check. The first successfully observed learned-mass transition was dense uncollapsed NUTS (initial step 0.1) at sweep 101; its 101–110 chunk had zero divergences but 10/10 tree caps. Continue the remaining checks before drawing any structure comparison.

All nine collapsed-NUTS tasks have now failed before the first learned mass is used: initial step 1.0 committed 40 sweeps, 0.1 committed 20, and 0.01 committed 70, for each mass structure. Steps 1.0 and 0.01 failed conditional-GP geometry audits; 0.1 failed the aggregate invalid-state predicate. Therefore no learned collapsed-NUTS mass comparison is available. All nine uncollapsed runs validated use of nonidentity mass at sweep 101; their 101–110 chunks each had zero divergences and 10 tree caps. MH/MALA/MMALA have completed 200; uncollapsed NUTS remains running.

Final 200-sweep results: all 21 sampling tasks terminated. MH/MALA/MMALA and all nine uncollapsed-NUTS combinations completed exactly 200 warmup sweeps (exit 0). All nine collapsed-NUTS combinations failed (exit 1) before the first mass update. Every run used the same original model data, fit and chain-0 initial state; per-method keys were reused across tuning variants, so these are paired tuning checks, not independent replicate chains. No sampling retries, new jitter, statistical-model changes, or production jobs were introduced.

| Method | Mass | Initial NUTS step | Committed sweeps | Slurm elapsed (s) | Committed sweep driver (s) | Peak RSS (MiB) | Outcome |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| mh | — | — | 200 | 49 | 26.540 | 560.9 | COMPLETED |
| mala | — | — | 200 | 243 | 221.177 | 676.2 | COMPLETED |
| mmala | — | — | 200 | 961 | 763.056 | 931.3 | COMPLETED |
| nuts | diagonal | 1 | 200 | 1217 | 1196.196 | 699.9 | COMPLETED |
| collapsed_nuts | diagonal | 1 | 40 | 390 | 284.256 | 699.1 | FAILED |
| nuts | diagonal | 0.1 | 200 | 1232 | 1206.458 | 683.1 | COMPLETED |
| collapsed_nuts | diagonal | 0.1 | 20 | 233 | 149.815 | 670.4 | FAILED |
| nuts | diagonal | 0.01 | 200 | 1255 | 1229.232 | 699.6 | COMPLETED |
| collapsed_nuts | diagonal | 0.01 | 70 | 648 | 540.185 | 705.9 | FAILED |
| nuts | kronecker | 1 | 200 | 1251 | 1224.412 | 742.5 | COMPLETED |
| collapsed_nuts | kronecker | 1 | 40 | 405 | 296.423 | 718.0 | FAILED |
| nuts | kronecker | 0.1 | 200 | 1222 | 1194.087 | 738.5 | COMPLETED |
| collapsed_nuts | kronecker | 0.1 | 20 | 223 | 142.947 | 705.3 | FAILED |
| nuts | kronecker | 0.01 | 200 | 1242 | 1215.208 | 735.6 | COMPLETED |
| collapsed_nuts | kronecker | 0.01 | 70 | 621 | 512.023 | 684.8 | FAILED |
| nuts | dense | 1 | 200 | 1209 | 1182.070 | 738.9 | COMPLETED |
| collapsed_nuts | dense | 1 | 40 | 288 | 204.181 | 897.7 | FAILED |
| nuts | dense | 0.1 | 200 | 996 | 968.881 | 941.3 | COMPLETED |
| collapsed_nuts | dense | 0.1 | 20 | 164 | 102.744 | 902.7 | FAILED |
| nuts | dense | 0.01 | 200 | 1191 | 1167.884 | 746.5 | COMPLETED |
| collapsed_nuts | dense | 0.01 | 70 | 480 | 394.235 | 733.1 | FAILED |

Slurm elapsed includes startup, initialization, compilation, checkpoint I/O and uncommitted failed work; committed sweep time excludes failed chunks. MaxRSS comes from the sampling job step. Each task had four CPUs and 16 GiB on Cascade Lake nodes u086/u087/u088/u117/u118; different nodes and concurrent workloads limit direct timing comparisons. Allocation cost was 62,080 core-seconds (17.2444 core-hours) for all sampling attempts, including failures, plus 664 core-seconds for tests/preparation: 62,744 core-seconds (17.4289 core-hours) total. Scheduler wait is separately recoverable from Submit/Start in `accounting.psv`; dense-step-0.01 tasks started later. No job-step allocation rows were double-counted.

MH variance remains 1e-6; MALA's initial preconditioner remains identity. Accepted-site proportions over 200 sweeps: MH 3,644/12,000 = 30.367%; MALA 6,865/12,000 = 57.208%; MMALA 6,903/12,000 = 57.525%. MH's halves were 41.4% and 19.333%, MALA's 53.75% and 60.667%, MMALA's 57.567% and 57.483%. MH/MMALA had no unmoved sites in either half; MALA had one unmoved site, also unmoved across all 200. These are movement/acceptance checks, not convergence or ESS evidence.

| Uncollapsed NUTS mass | Initial step | Divergences / 200 | Tree caps / 200 | Divergences / last 100 | Tree caps / last 100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| diagonal | 1 | 9 | 192 | 1 | 99 |
| diagonal | 0.1 | 12 | 189 | 1 | 99 |
| diagonal | 0.01 | 9 | 193 | 2 | 99 |
| kronecker | 1 | 9 | 192 | 1 | 99 |
| kronecker | 0.1 | 12 | 189 | 1 | 99 |
| kronecker | 0.01 | 8 | 193 | 1 | 99 |
| dense | 1 | 11 | 190 | 3 | 97 |
| dense | 0.1 | 13 | 188 | 2 | 98 |
| dense | 0.01 | 8 | 193 | 1 | 99 |

Every successful uncollapsed NUTS run first used learned nonidentity mass at sweep 101. Mass structures were checked against checkpoint metadata (`nuts.mass_matrix`, eta-v1 coordinates); final inverse masses have positive finite eigenvalues. The second window closes at sweep 150; adaptation is still in progress at 200 and no production tuning is frozen. All structures retain 97–99% tree-cap frequency in the latter half, so this short check does not establish a successful mixing improvement or a preferred NUTS configuration.

Collapsed-NUTS outcomes are identical by mass label at these committed boundaries: initial step 1.0 saved 40 sweeps, then failed the field-site audit of the 41–50 chunk with `Unjittered C_f_given_s is singular or numerically singular`; step 0.1 saved 20, then failed within 21–30 with `Gibbs sweep produced invalid model state`; step 0.01 saved 70, then failed the audit of 71–80 with the same conditional-GP singularity message. The state predicate does not identify the precise bad component or sweep. Only committed valid chunks are summarized: respective divergence/tree-cap counts are 10/30 (40 draws), 5/15 (20 draws), and 12/58 (70 draws). No run used learned mass before failing, so no collapsed-target learned-mass comparison can be claimed.

Validation: seven modified-mass tests passed on Unity; fresh preparation preserved all 34 numerical arrays bitwise and validated all four starts and 21 factories. Final audit verified all nine archive hashes, every committed draw-chunk checksum, contiguous sweep numbers, finite stored states/full joint densities, check limit 200 with the unchanged 1,000-sweep schedule, warmup checkpoint phase, expected mass/eta metadata, successful exits/empty stderr for the twelve completions, failed exits/error logs for the nine failures, and zero production clocks. Current source hashes still match the archived run implementation, so no code changed during execution. Final mass eigenvalues are positive for all saved NUTS boundaries. Shell syntax and `git diff --check` passed. The completed-run audit does not turn failed runs into successful ones.

Evidence: `results/unity-mass-200-20260928/summary.json` (all 21 records, per-half/last-20 diagnostics, time/memory, final mass shape/eigenvalues and errors), `accounting.psv`, `manifest.json`, `submissions.json`, `validation.out`, and all per-variant draw/checkpoint/log files. `prepare.py`, `summarize.py`, and `progress.py` preserve the execution/audit workflow. The user queue is empty. Full production remains unsubmitted.

### Configurable blocked NUTS (2026-09-28; awaiting blocking decisions)

The user requested a configurable NUTS block size, giving `12 × 5` as an example. Clarification is pending whether 12 counts sites (60 sites → five blocks, 36 theta coordinates per block because d=3) or scalar theta coordinates. Proposed implementation, awaiting explicit confirmation: contiguous site-major blocks in fixed sequential order, covering every site once per outer sweep; both collapsed and uncollapsed targets; separate step size and mass/window adaptation for every block; diagonal/dense/Kronecker choices apply within each block (Gamma_site is block-site × block-site and Gamma_param remains d × d). The conditional target must keep all outside-block eta at their latest values and recompute density/gradient at each block entry; it must not replace the full spatial/GP target with independent site models. Preserve the existing outer Gibbs schedule and immediate c_f refresh after the entire theta block sweep. No numerical/model changes or new cluster runs have been performed for this task.

Inspection shows the current NUTS kernel, chain tuning/adaptation, scalar diagnostics, chunk runner and checkpoint protocol all assume a single all-site trajectory. Implementation must update them together after the decisions: explicit block specification and PRNG splitting, per-block diagnostics/tuning, frozen adaptation, restart serialization and comparison configuration. Preserve the existing one-block behavior when block size is absent or covers all sites. Proposed handling of a non-dividing size is one smaller final block, without padding or dropping sites; validate a positive integer no larger than the site count. Acceptance work should cover direct sequential BlackJAX conditional references (including dependence on earlier blocks), both targets and bounds modes, all mass structures, a smaller final block, one-block equivalence, complete outer Gibbs ordering, and bitwise warmup/frozen/restart behavior. These validation details are implementation proposals, not evidence of completed work.


### 2026-09-29 — Review of MH/MALA/MMALA production attempt 11193199

The user requested inspection of `results/unity-production-20260929`. Contrary to the earlier production-deferred status above, this archive records an actual submitted attempt: three methods × four chains, 1,000 warmup sweeps, six production compute-hours per chain, chunks of ten. All twelve sessions failed; none completed the production budget. This review does not complete Stage 14b. No sampling, retries, tuning changes, or model changes were performed.

| Method | Chain (zero-based) | Last committed sweep | Production draws | Production compute minutes | Production accepted-site % | Failure |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| MH | 0 | 15350 | 14350 | 24.237 | 3.611 | Conditional-GP singularity |
| MH | 1 | 1600 | 600 | 0.936 | 3.644 | Conditional-GP singularity |
| MH | 2 | 870 | 0 | 0 | — | Invalid Gibbs state |
| MH | 3 | 5370 | 4370 | 8.575 | 3.995 | Invalid Gibbs state |
| MALA | 0–3 | 990 each | 0 | 0 | — | Zero empirical covariance at warmup sweep 1000 |
| MMALA | 0 | 6480 | 5480 | 297.837 | 57.169 | Conditional-GP singularity |
| MMALA | 1 | 70 | 0 | 0 | — | Invalid Gibbs state |
| MMALA | 2 | 2090 | 1090 | 63.921 | 57.459 | Conditional-GP singularity |
| MMALA | 3 | 3390 | 2390 | 126.375 | 57.506 | Conditional-GP singularity |

The open task-0 stderr is MH chain 0: `Unjittered C_f_given_s is singular or numerically singular`, raised in the end-of-chunk field-site audit for sweeps 15351–15360. Other failures occur in the next ten-sweep chunk after the saved boundary; the aggregate invalid-state check does not identify the exact sweep or invalid component. MALA's warmup failure diagnostics record 1000 attempted sweeps, but the rejected final chunk leaves only 990 committed. Its never-moved/zero-covariance sites (zero-based) are chain 0: [27]; chain 1: [29,30,31]; chain 2: [27,28,29,31,33]; chain 3: [27,28,30].

Acceptance rates summarize retained sampling sweeps, not convergence. MALA has no production draws; MH/MMALA each have only three prematurely terminated production chains with unequal lengths. No successful four-chain convergence or ESS/sec comparison is established. The six-hour budget counts production sweep computation, excluding warmup/compilation/I/O; MMALA chain 0 used 297.837 production minutes despite 435.122 total wall minutes. Recorded I/O is substantial in some sessions, so wall-time comparisons require care.

Validation: `audit_results.py` verified all 3,918 committed chunk SHA256 checksums, contiguous iteration ranges and phase labels, finite stored model states/full joint densities, checkpoint/session iteration agreement, and exact agreement of recomputed acceptance/movement counts with sessions. No uncommitted draw files remain. This is artifact validation, not a fresh numerical covariance audit of every draw. All twelve stderr files report Python errors and srun exit code 1. Slurm accounting could not be queried because the scheduler/database connection was unavailable; no scheduler-level elapsed/RSS claim is made. Reproducible audit and full chain records are saved as `results/unity-production-20260929/audit_results.py` and `audit-summary.json`.

Next task: localize the conditional-GP degeneracy/invalid state and diagnose MALA's immobile sites before another production attempt. Numerical/model remedies remain undecided and require explicit approval where they change the specified model or numerical policy. Existing unrelated code and plan edits were preserved.

### 2026-09-29 — Blocked NUTS 300-sweep checks submitted; no monitoring requested

The user explicitly requested submission with block size 12, varied initial step sizes, both collapsed/uncollapsed targets, and termination of monitoring immediately after submission. Submitted six chain-0 checks using the existing implementation's site-based block semantics: 60 sites split into five contiguous 12-site blocks, 36 eta coordinates per block. Use initial steps 1.0/0.1/0.01 from the previous comparison, current default diagonal mass, and the established 1,000-sweep warmup schedule with a 300-sweep check limit. Each outer sweep visits all five blocks. This request uses existing blocked behavior; no sampler/model implementation was modified. No production is requested.

Submission receipts: validation/preparation job `11195750`; step 1 array `11195751`; step 0.1 array `11195752`; step 0.01 array `11195753`. Each array has tasks 12 (uncollapsed chain 0) and 16 (collapsed chain 0), with `afterok:11195750` and `--kill-on-invalid-dep=yes`. Validation runs `test_nuts_blocks` then imports frozen numerical data/fit/starts from the previous production prepared archive, verifies bitwise preservation, validates all six factories, and records new implementation hashes. Validation results are pending, not claimed as passed. All jobs use stat-cascade/stat-users, Cascade constraint, four CPUs and 16 GiB; validation limit 30 minutes and each sampling limit three hours to accommodate five block transitions per sweep.

Local preparation validation: Python compilation of preparation/submission scripts and shell syntax check passed. The initial sandboxed scheduler call failed before returning any job ID; the authorized network-enabled submission returned all four IDs successfully. Exact commands/receipts are in `results/unity-blocked-300-20260929/submissions.json`; validation logs and per-step sampling logs remain under that directory. Original experiments/configuration and prior artifacts are unchanged. No queue/progress/result polling was performed after successful submission, as requested. Next task is user-initiated result review; stop work now without monitoring.


### 2026-09-29 — Last committed production-state GP geometry
The user requested minimum eigenvalues of C_f_given_s and pairwise theta distances immediately before the MH/MALA/MMALA failures. The available states are the last committed checkpoints; failed ten-sweep chunks were not serialized. No transitions were replayed, and these results do not claim to reconstruct the actual failing state. All coordinates are unbounded, so eta equals standardized theta. Compute the unjittered site-level conditional correlation C_ff - C_fs solve(C_ss,C_fs.T), symmetrized exactly as the host audit; also independently evaluate the sampler Cholesky-solve expression. No coefficient-variance Kronecker factor is included in this C_f_given_s.
| Method | Chain | Saved sweep | Minimum eigenvalue (host audit) | Minimum / audit threshold | Minimum standardized field distance | Closest sites (zero-based) | Minimum kernel-metric field distance |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| mh | 0 | 15350 | 2.066444e-15 | 1.394 | 0.101755 | [17, 45] | 0.0238401 |
| mh | 1 | 1600 | 2.800290e-14 | 172.153 | 0.123619 | [55, 59] | 0.0246699 |
| mh | 2 | 870 | 1.225807e-14 | 8.413 | 0.0612664 | [0, 1] | 0.00981567 |
| mh | 3 | 5370 | 9.637163e-16 | 2.525 | 0.0241012 | [18, 38] | 0.00400804 |
| mala | 0 | 990 | 2.451038e-14 | 15.926 | 0.0934976 | [17, 45] | 0.0194088 |
| mala | 1 | 990 | 7.384396e-14 | 464.639 | 0.106581 | [58, 59] | 0.0191091 |
| mala | 2 | 990 | 1.581025e-14 | 10.629 | 0.0854521 | [43, 44] | 0.0128672 |
| mala | 3 | 990 | 7.267688e-15 | 19.467 | 0.157659 | [57, 58] | 0.0300738 |
| mmala | 0 | 6480 | 2.901246e-15 | 1.896 | 0.0684695 | [1, 21] | 0.0130871 |
| mmala | 1 | 70 | 4.550992e-14 | 283.431 | 0.122575 | [56, 59] | 0.0206932 |
| mmala | 2 | 2090 | 3.135389e-14 | 17.994 | 0.102611 | [23, 24] | 0.0165547 |
| mmala | 3 | 3390 | 1.067508e-15 | 2.865 | 0.137506 | [18, 38] | 0.0222391 |

All twelve saved states pass the relative host threshold 32*eps64*lambda_max in this recomputation, consistent with their committed status. MH chain 0, MH chain 3, MMALA chain 0, and MMALA chain 3 have minimum/threshold ratios 1.39, 2.52, 1.90, and 2.86 respectively. The corresponding matrices are already very ill-conditioned. No exact field-field or field-library theta coincidences are present. Kernel-scaled nearest field distances span approximately 0.00401–0.0301; nearest field-library distances span 0.0379–0.0986. Thus exact pair collisions do not explain these stored states; near-dependence of the smooth conditioned GP can occur without exact duplicates. This is a geometric observation, not proof of the source of each failed transition or of MALA's zero-covariance adaptation failure.

Validation/evidence: `inspect_geometry.py` completed for all twelve checkpoints with float64 JAX kernel values, NumPy host eigensolves and a second JAX Cholesky-solve covariance calculation. Both covariance expressions have positive computed minima for these checkpoints; their small differences at approximately machine precision materially affect minima near 1e-15, so last digits are not robust across numerical implementations. `checkpoint-geometry.json` records checkpoint SHA256 hashes, both minima, lambda_max, exact thresholds, covariance differences, distance quantiles, nearest pairs and fixed GP length scales. `checkpoint-geometry.npz` stores all twelve theta arrays, both conditional matrices/eigenspectra, and complete standardized/kernel-scaled field-field and field-library distance matrices. Distances exclude the field self-diagonal. Physical-coordinate Euclidean distances are not pooled because parameter coordinates have different scales/units. No model/sampler changes or cluster sampling jobs were made. Next task, if requested, is explicit replay/instrumentation of failed chunks to localize the first invalid state; that has not been performed here.

### 2026-09-29 — CV–NLPD library length-scale selection

The user requested the CV–NLPD criterion already present in the current `Preprocessing and hyperparameter selection.md` as an alternative GP hyperparameter selection method and requested a fit on the existing synthetic library. `gp.cv_nlpd` now computes the stated $r$ foldwise Gaussian scores with zero mean and a separate $r-1$ divisor for every training-fold output variance. `fit_library_length_scales(method="cv_nlpd")` minimizes their average over log length scales, using the same explicit starts, tolerances, optional bounds, unjittered singularity checks, fixed jitter policy, SciPy optimizer and attempt records as profile fitting. `comparison.prepare` already forwards the `library_fit` mapping; `comparison.validate_config` now accepts `profile` (the unchanged default) or `cv_nlpd`. The existing `experiments/comparison.json` and frozen Stage 14b prepared archive retain profile fitting; no checkpoint or calibration output has been changed. Because preparation records implementation hashes, loading that old prepared archive under the modified source requires a fresh preparation or the original frozen code version.

On the Stage 14a loading-only library ($r=20,d=3,k=5$), with float64, zero jitter, no log bounds, starts $(-1,-1,-1),(0,0,0),(1,1,1)$ and the prior profile optimum, `gtol=1e-6`, `ftol=1e-10`, `maxiter=1000`, all four starts converged to the same neighborhood. The best CV–NLPD length scales in standardized coordinates are $(3.0685738163,4.7534126838,5.2175510188)$, log length scales $(1.1212128987,1.5588628198,1.6520281383)$, and mean NLPD $20.8884035055$. Evaluating the existing profile scales $(3.35581539,6.82735774,6.89863374)$ under CV–NLPD gives $22.0280765039$. Full-library $q_\ell/r$ diagnostics at the CV optimum are approximately $(3384.51,209052.42,5993059.64,21549109.27,35490471.40)$; these are not substituted for sampled $\sigma_{c,\ell}^2$.

The CV-fit unjittered $C_{ss}$ has minimum eigenvalue $3.9712\times10^{-6}$ and condition number $4.27\times10^6$, versus $9.7058\times10^{-7}$ and $1.83\times10^7$ for the profile fit. On the four *initial states* stored in the old prepared archive, the new length scales pass the existing unjittered $C_{f\mid s}$ check: minimum eigenvalues are approximately $(1.93,7.63,4.10,1.06)\times10^{-13}$, versus $(0.0773,0.424,0.245,0.0679)\times10^{-12}$ at the profile fit. These are still very small eigenvalues, and initial-state checks cannot establish that future MCMC proposals or failed production states remain nonsingular. The failed ten-sweep chunks were not saved locally, so no claim of resolving those failures is made.

Validation: eight GP tests and two focused comparison tests passed, including an independent SciPy foldwise multivariate-normal density reference, finite-difference JAX gradient check, multistart minimization, zero-variance fold rejection, unchanged profile-fit tests, method configuration validation and preparation metadata regression. `git diff --check` passed. Next task, if a new calibration run is desired, is to prepare a new frozen archive with `library_fit.method="cv_nlpd"`, validate its initial states and evaluate GP geometry during a short execution check before any production run. Existing run artifacts must remain tied to their original fitted length scales.

### 2026-09-29 — CV–WMSE length-scale calculation

The user identified `sum(residual**2 / predictive_variance, axis=1)` as a possible WMSE selection criterion and requested the associated length scales. The implemented `cv_wmse` is the mean of exactly this per-run quadratic term, using the same leave-one-run-out predictive means and training-only output variances as CV–NLPD. The shared fold calculation is factored into a pure JAX helper to keep both scores on identical predictions and variances. `fit_library_length_scales(method="cv_wmse")` and `library_fit.method="cv_wmse"` expose it as an explicit alternative. Profile remains the default; the current comparison configuration and archived prepared inputs remain unchanged.

On the same Stage 14a library ($r=20,d=3,k=5$), with zero jitter, no log bounds, `gtol=1e-6`, `ftol=1e-10`, and `maxiter=1000`, a first seven-start search revealed multiple local minima and some optimizer paths that reached singular unjittered $C_{ss}$; those paths were rejected. A deterministic 27-start grid over log scales $\{-2,0,2\}^3$ improved the best WMSE to $1.0309552539192202$. A further 64-start grid over $\{-3,-1,1,3\}^3$ found no lower value; 60 starts converged and four had singular trial paths. The best observed length scales in standardized coordinates are $(0.2014591522,3.2209765010,5.6345367281)$, log scales approximately $(-1.6021686368,1.1696845748,1.7289149308)$. This is a multistart numerical optimum, not a proof of a global minimum.

At these scales, mean CV–WMSE is $1.0309552539$ and mean CV–NLPD is $34.8147049119$. For comparison, the CV–NLPD fit has WMSE $8.4091489660$ and NLPD $20.8884035055$, and the original profile fit has WMSE $13.8674391561$ and NLPD $22.0280765039$. Thus the quadratic-only criterion produces a substantially worse predictive-density score. The unjittered $C_{ss}$ minimum eigenvalue is $0.0273984$ and condition number $103.73$. On the four old prepared *initial states* only, the WMSE scales pass the existing $C_{f\mid s}$ check, with minimum eigenvalues approximately $(3.69\times10^{-5},1.97\times10^{-5},1.01\times10^{-5},5.44\times10^{-6})$. These checks do not establish validity throughout MCMC.

Validation: the independent NumPy/SciPy fold reference now also checks the exact WMSE quadratic term; a finite-difference check verifies its JAX gradient, and the multistart fit/configuration tests cover the new method. Eight GP tests and two focused comparison tests passed; `git diff --check` passed. No new prepared archive or calibration chain was created. Before adopting WMSE for a calibration comparison, prepare and validate new frozen inputs and assess predictive uncertainty as well as field-conditional GP geometry; a lower WMSE alone is not evidence of better calibrated predictions.

### 2026-09-29 — Scores at Yingyu's reported short length scales

The user reported Yingyu's $(0.65,0.9,1.35)$ and asked how WMSE/NLPD behave near one. The archived `Archive/Yingyu Model.md` gives the same squared-exponential formula in standardized $\widetilde\theta$ but does not record those numerical values or fully establish identical data preprocessing/standardization. The following comparison therefore treats the user-provided vector as length scales in *this project's* Stage 14a sample-standardized coordinates; it is not a claim to reproduce Yingyu's own fit.

| Length scales | Mean CV–WMSE | Mean CV–NLPD | Mean fold log predictive determinant |
| --- | ---: | ---: | ---: |
| $(0.65,0.9,1.35)$ | 1.9624887313 | 36.1759420697 | 61.2000100760 |
| $(1,1,1)$ | 1.9623169115 | 35.0366739979 | 58.9216457522 |
| CV–WMSE fit $(0.20146,3.22098,5.63454)$ | 1.0309552539 | 34.8147049116 | 59.4090692372 |
| CV–NLPD fit $(3.06857,4.75341,5.21755)$ | 8.4091489662 | 20.8884035056 | 24.1782727130 |
| Profile fit $(3.35582,6.82736,6.89863)$ | 13.8674391648 | 22.0280765071 | 20.9993285174 |

Thus Yingyu's short vector has WMSE about 1.90 times the best observed WMSE but far below the WMSE at the profile fit. Its NLPD is about 15.29 higher per held-out five-coefficient vector than the CV–NLPD fit. The main difference is the fold log predictive determinant: $61.20$ versus $24.18$; the WMSE term itself is smaller for Yingyu's vector. A common-factor sweep of Yingyu's vector at factors $0.5,1,1.5,2,3$ gave NLPD $41.36,36.18,31.64,28.40,24.94$ respectively, while WMSE was $3.80,1.96,2.13,3.26,6.82$. No global monotonicity is claimed. Independently recomputed the Yingyu, unit and CV–NLPD rows using NumPy/SciPy Cholesky solves; they agree with the JAX score values. No fitting code or run artifacts were changed. Next task, if the Yingyu vector is intended for a matched comparison, is to establish its exact coordinate standardization and preprocessing before attributing score differences to hyperparameter selection alone.

## 15. Sampler-focused refactor and staged validation review (2026-09-30)

Status: R1–R3 complete; local R4 validation passed, adequate production comparison pending. The two aims remain making the paper's sampler easier to read by removing unrelated responsibilities, and localizing sampling problems through staged validation. Fresh-run reproducibility, infrastructure removal and CLI/configuration responsibilities below are explicit user decisions. Implementations and evidence are recorded below. Numerical remedies and changes to the model, target, sampler, adaptation, or failure policy remain outside this refactor.

### Reproducibility decision

The user clarified: "Exact historical reproduction is not a goal. The public implementation should reproduce the model, sampling algorithms, and experimental setup sufficiently to regenerate comparable results from a fresh run." This supersedes this review's earlier proposal to recover historical source/environments and replay archived failures before moving code. Old checkpoint compatibility, historical artifact regeneration and bitwise agreement with old runs are not acceptance requirements.

The public workflow must document and execute input preparation, frozen standardization, library length-scale selection, target construction, initialization, warmup, retained sampling and analysis from fresh inputs. Record the specified priors/support, preprocessing/noise recipe, fitting starts/tolerances, spatial/GP settings, sampler blocking and adaptation, numerical policy, seeds, chain count, run budget, environment and diagnostic/reporting criteria. Keep essential scientific experiment settings explicit while separating cluster/job management from the core. The workflow must not require private absolute note paths, old prepared archives or historical Unity jobs; document input sources/formats or the synthetic-data recipe and method settings in the repository.

Within a fresh comparison, share the newly prepared data, frozen standardization/length scales, priors and numerical policy across methods, and match initial model states by chain index while using independent method/chain PRNG streams. These common inputs are generated for the new experiment; they need not equal historical prepared arrays.

Comparable results means validated agreement with the specified posterior and algorithms, and, where adequate valid draws exist, agreement of predeclared posterior/predictive summaries within assessed Monte Carlo uncertainty under the same experiment protocol. It does not require matching individual draws, runtime on different hardware, or the exact failure point of an old run. Fresh numerical failures must be reported and localized; this clarification does not resolve GP degeneracy or authorize a change to the target.

### Public scope and option interface

The user decided: "Checkpoint/restart, historical hash compatibility, and general-purpose job orchestration are outside the public implementation and will be removed rather than refactored." Remove their public code paths, dependencies, CLI modes and associated infrastructure-only tests; do not retain them in another public module. This includes restart serialization/loading/migration, code-hash compatibility gates, task-ID/job-array mapping, scheduler submission helpers, worker locks, committed-chunk recovery, resumable budget accounting and scheduler/session machinery. Existing numerical/reference coverage embedded in mixed infrastructure tests must be retained as fresh-run tests. Historical result files remain evidence of prior work and are not inputs to the public workflow.

Retain the small research execution path: prepare inputs, initialize, warm up, sample, record draws and diagnostics, and analyze. A declared run length or time budget, explicit chain PRNG streams, JIT function reuse and numerical batching are ordinary local execution concerns; they do not require a job framework or resumable clock. Any batching keeps the established numerical checks and update order. Simple result/configuration output and focused failure evidence remain useful, but cannot be loaded as a restart state. Validation must not depend on checkpoint machinery.

The user also decided: "Run-level options are exposed through the CLI, while sampler-specific options are defined in configuration files and can be overridden from the CLI when needed."

| Option responsibility | Public interface |
| --- | --- |
| Run-level setup | CLI options for input/output locations, method, seed, chain count, warmup length, retained-sampling length or budget, diagnostic mode and sampler configuration path; document defaults and compatible combinations |
| Sampler-specific settings | Sampler configuration files contain the declared settings relevant to the selected method: initial proposal/preconditioner, initial covariance-estimation period, epsilon/epsilon_G, target acceptance, NUTS mass structure/block size, step size and tree/divergence limits as applicable |
| Sampler overrides | Explicit, named CLI overrides take precedence over the selected sampler file; unspecified CLI options leave its values intact |
| Shared scientific specification | Keep model, preparation and library-fit settings explicit and common across methods; do not duplicate or change them through sampler configuration |
| Recorded experiment | Save the resolved run-level arguments and sampler settings with ordinary outputs, including applied overrides and implementation/environment information; no historical compatibility check is involved |

Use the existing JSON format for sampler files unless implementation reveals a concrete need for another format. Use a small parser and direct settings validation, not a generic configuration framework. Resolve sampler options once before initialization, validate the effective settings, and use those values throughout warmup and sampling. Reject unknown or method-inapplicable settings instead of silently ignoring them. CLI parser defaults must not overwrite file settings when an override is absent. Preserve all currently approved algorithm defaults and regularization; changing their exposure does not change their scientific meaning.

Validate combined run and sampler settings after applying overrides, including `num_initial <= num_warmup`, a valid NUTS block size for the prepared site count, positive step/proposal scales and supported mass choices. Preserve the distinction between an absent override and an explicit value such as all-site blocking; a user must be able to replace a file's blocked setting explicitly. File-only settings and file-plus-CLI settings that resolve to the same values must reach the same numerical kernel/adaptation configuration. Include these cases in interface tests. Numerical batch/audit spacing, if exposed, is a run-level CLI setting rather than a checkpoint or scheduler setting.

### Findings and proposed boundaries

The pre-refactor review found execution management concentrated in `mcmc.py` (1,218 lines), checkpoint `run.py` (535 lines) and experiment-management `comparison.py` (454 lines), with an empty README and greeting-only entry point. The implemented public path now connects prepared arrays and fixed specification to target/conditionals, theta transitions, exact coefficient refresh, warmup, retained draws and analysis. The following boundaries motivated the implemented module separation.

| Responsibility | Proposed treatment |
| --- | --- |
| Slope matching, spline/QR projection and frozen library standardization | Retain the specified preparation mathematics; separate source-specific CSV/XLSX readers, synthetic noise/truth, plotting and archive CLI from it. Prepared numerical inputs must remain reproducible and evaluation truth must stay outside the model. |
| Site transformations/Jacobian, GP conditioning, priors/targets, Gibbs conditionals, theta kernels | Keep as the mathematical core, preserving source notation, shapes and site-major stacking. Transformations are part of the target, not optional execution machinery. |
| Adaptation formulas, schedule and production freeze | Retain as sampler algorithm code. Keep validation distinct from those formulas and remove restart serialization; preserve approved zero-scatter, step-size, mass and block policies. |
| Complete Gibbs sweep and small warmup/production loops | Keep the seven updates visible in one place, with model state, tuning, adaptation and diagnostics distinct. Separate model validation from sampler validation rather than constructing a surrogate random-walk chain to validate other samplers. |
| JIT reuse and local execution | Retain only the simple sweep/warmup/production loop and needed compilation reuse or batching; keep the numerical core usable without output paths or Slurm configuration. |
| Checkpoint/restart, historical hash compatibility and job orchestration | Remove from the public implementation, including infrastructure-only dependencies and tests; preserve scientific tests and historical results. |
| CLI, sampler configuration and experiment recipe | Keep a thin fresh-run entry point with run options on the CLI, sampler files and explicit overrides, plus simple draw/diagnostic/configuration output. |

Prefer a few functions and responsibility-based modules over a universal sampler class hierarchy, callback framework or generalized configuration/caching system. Preserve currently implemented sampler/bounds/branch/block/mass options; removing an option is a separate scope decision. The implemented module layout is recorded below in this plan.

### Validation layers and failure evidence

Existing mathematical/reference tests are divided by component. Before refactoring, the main missing capability was runtime context: combined state checks happened after all seven updates and `_checked_call` raised an unstructured string. Named update checks and structured host context now fill this gap. Normal geometry checks still inspect a numerical batch endpoint; diagnostic mode applies the same criterion to each completed sweep. Archived failures motivate this context but are not reconstructed.

| Layer | Normal execution | Additional diagnostic execution |
| --- | --- | --- |
| Preparation/initialization | Static shapes, dtype, finite inputs, support, frozen maps/factors and initial state checks | Independent preprocessing/GP/target references; input and implementation provenance |
| Each Gibbs update | Name the update and check its output finiteness; check positivity for sampled variances without extra matrix factorizations | Inspect conditional parameters and the existing factor/solve/draw for `delta`, `sigma_y2`, `mu_theta`, `Sigma_theta`, `sigma_c2`, `eta` and `c_f` |
| Theta transition | Attach site/block and current/proposal/selected roles to existing density, gradient, metric and proposal checks | Target components, gradient norms, forward/reverse drift/diffusion and log-proposal terms; preserve standard NUTS tree behavior |
| Adaptation | Identify step-size/covariance/mass failure and warmup completion gate, including site/block/window | Covariance/mass spectra, acceptance versus movement, exact zero-scatter counters and quantities used for the next transition |
| Numerical batch boundary | Preserve the existing one geometry audit per numerical batch; identify the audited endpoint and batch range; report an invalid batch without recording it as valid draws | Apply the same geometry criterion to completed states in a fresh diagnostic run, independently of the first invalid update |
| Retained-chain analysis | Keep convergence and efficiency assessment separate from numerical execution success | Multiple-chain R-hat/ESS/MCSE, posterior agreement and predictive checks after adequate valid draws exist |

An error record should identify method/chain, phase, exact sweep or audited batch endpoint, update, quantity, current/proposal/selected role, site/block when applicable, and the failed predicate. For a fresh run, retain focused context to diagnose the failure: relevant input/current values, key information, tuning/adaptation summaries, requested batch length, resolved scientific configuration, implementation version and numerical environment. A failure artifact is diagnostic evidence, not valid draws or a serialized state for resuming sampling. Avoid dumping large matrices every sweep; collect expensive evidence only in explicitly selected diagnostic execution. Failure localization uses fresh runs and injected errors and has no checkpoint/replay prerequisite.

The collapsed observation covariance can remain usable because it includes observation noise even when the GP coefficient prior factor is unusable. Thus a finite collapsed theta target does not establish that the subsequent `c_f` refresh or next coefficient-variance update can factor their required GP covariance. Diagnose those operations separately; this is a possible failure mechanism, not an established explanation for the archived failed transitions.

Preserve existing outcome distinctions: valid negative-infinity candidate targets can produce ordinary rejection; current-state density/gradient failure is fatal under the existing checked drivers; BlackJAX's candidate handling and finite-state NUTS divergences/tree caps remain algorithm outcomes. Do not turn every intermediate NUTS nonfinite trajectory value into a new fatal policy. Do not add eigensolves to every production proposal, infer new support from an eigenvalue threshold, or introduce jitter, clipping, redraws or retries through validation. Extra diagnostic checks must not feed back into proposal decisions or retained samples.

### Proposed sequence and acceptance gates

| Task | Work | Acceptance before completion |
| --- | --- | --- |
| R1 — public method, experiment and option contract | Record core inputs/outputs, model/target, update order, PRNG handling and numerical/failure policy; enumerate run-level CLI versus sampler-file settings and overrides; define the fresh-run recipe and reading order | Scientific settings and inputs are explicit; CLI/file precedence and resolved defaults are documented; independent mathematical/reference tests and fresh fixtures establish correctness without historical artifacts |
| R2 — failure context | Add small named update checks and host sweep context; retain focused failure evidence for fresh runs | Independent error injection identifies update/quantity/sweep/site/block; normal rejection/divergence handling and scientific transition behavior are preserved; diagnostic evidence cannot enter retained draws or alter sampling decisions |
| R3 — public simplification and interface | Remove checkpoint/restart, historical hash compatibility and general-purpose job orchestration; separate preparation I/O; keep formulas and sweep/warmup loops readable; implement the thin CLI and sampler-file overrides | Removed facilities have no public entry points/import requirements; mathematical identities, Gibbs conditioning/refresh, sampler/adaptation references and production freeze pass; CLI defaults preserve file settings, explicit overrides reach the actual sampler, and unknown/inapplicable options are rejected |
| R4 — fresh-run integrated validation | Exercise CLI preparation through analysis with fresh inputs/initialization, all implemented methods and relevant bounds/branch/block/mass cases, failed updates, warmup gates and batch-end audits | Focused integration and small known-posterior checks pass; documented commands regenerate inputs and outputs without checkpoints or compatibility gates; recorded effective settings match execution; production retains lightweight validation; adequate-chain comparisons use Monte Carlo uncertainty, while short checks establish only execution behavior |

R2 and R3 can proceed independently after R1 where their changes do not overlap; neither depends on reproducing an archived failure. Keep independent reference tests rather than replacing them with tests of the moved implementation itself. Remove infrastructure-only checkpoint/restart/hash/job tests while retaining any scientific assertions in mixed tests. Use justified numerical tolerances for deterministic density/gradient/conditional calculations and Monte Carlo uncertainty for stochastic posterior checks. Bitwise comparisons can remain useful local regression checks for unchanged numerical kernels, but are not the public reproducibility criterion. Identical seeds do not imply identical trajectories across implementations, compilation settings or backends.

### Fresh-run artifacts and outstanding decisions

The public workflow generates new prepared inputs, ordinary draw/diagnostic outputs and effective configuration records. Checkpoint formats, restart state loading, historical implementation-hash compatibility and migration are removed from public scope. Preserve existing results as historical evidence without relabeling them as new results. Record scientific configuration, input provenance, implementation version and numerical environment so fresh results can be interpreted and compared; these records document a run and do not enforce compatibility with old artifacts.

No new model choice was made. GP degeneracy remedies, retuning, pruning sampler features, and any change in numerical/failure policy remain separate decisions. Automatic bounded initialization is still unspecified: bounded transformations, priors and kernels remain available through explicitly supplied complete states in the core API; the CLI's declared automatic recipe is unbounded and rejects a bounded request rather than inventing initialization. R1–R3 local acceptance gates pass. An adequate production comparison remains outstanding and requires a separate execution decision after fresh failures are understood. Exact historical reproduction is not a prerequisite.

Initial plan-review evidence (before implementation): inspected the implementation, runtime checks and relevant reference/integration tests; checked the four primary notes' preparation, coordinates, GP and sampling specifications against these boundaries; independently reviewed architecture and failure localization. At that review stage only this plan was edited, no tests/transitions were executed and `git diff --check` passed. The later implementation evidence follows below.

Clarification update (2026-09-30): revised scope, task order, validation gates and artifact policy to implement the user's fresh-run reproducibility decision. Historical replay/environment recovery and old-run bitwise matching are no longer required. This update changes the plan only; no public workflow, refactor or fresh experiment has been implemented or executed.

Public-scope update (2026-09-30, before implementation): recorded the user's decision to remove checkpoint/restart, historical hash compatibility and general-purpose job orchestration, replacing the earlier optional-checkpoint/refactor proposal. Recorded run-level CLI settings, sampler configuration files, explicit override precedence and interface validation gates. That update changed only the plan; the subsequent implementation is recorded below.

### Implemented public path and acceptance record

The user subsequently authorized this plan ("이 계획대로 진행하자"). Implementation is limited to the public refactor and local validation; no historical result files, old data/checkpoints, external notes or cluster state were changed. Existing user changes to CV–NLPD/CV–WMSE GP fitting and GP tests were preserved.

| Task | Implemented result | Current acceptance evidence |
| --- | --- | --- |
| R1 — complete | README documents input formats/shapes, site-major stacking, fixed scientific JSON, library fitting, seven-update order, initialization, independent sampling streams, option precedence and analysis semantics | Four primary notes reread; independent mathematical/reference fixtures passed; package defaults and current float64 CPU environment documented |
| R2 — complete | `validation.py` separates model/key/SPD checks and `SamplingError`; sweeps name all seven outputs; theta checks label roles/sites/blocks; host checks add method/chain/phase/sweep/batch and warmup context | 17 staged-error/hotpath tests passed, including attempted invalid mass/proposal evidence, exact latest Gibbs conditioning and normal-versus-diagnostic trajectory equality; independent review found no remaining material error after diagnostic corrections |
| R3 — complete | Removed `run.py`, `comparison.py`, combined job config and Unity shell helpers; `data.py` holds projection mathematics, `preprocessing.py` source/noise/truth/plot I/O; `experiment.py` and `cli.py` expose fresh execution; packaged method JSONs resolve once; `SweepRunner` has no I/O | Full suite passed; CLI all-method preparation/warmup/retention, actual override use, file/default preservation, unknown/inapplicable settings, direct-driver equivalence, timed budget and failure outputs passed; wheel/sdist and installed-package checks passed; lock resolves 23 packages |
| R4 — local checks passed, production comparison pending | Fresh CLI integration plus independent posterior/reference tests cover bounds, branches, mass/block variants, refresh, warmup freeze and numerical batching | Known-posterior/reference checks passed; supplied-input preparation passed and a deliberately short warmup correctly triggered its freeze gate; adequate production posterior/predictive agreement is not established |

Reading order: numerical `data.py` → `state.py`/`transforms.py`/`gp.py` → `targets.py`/`gibbs.py` → `samplers/*` → `adaptation.py` (including NUTS mass estimation) → complete sweeps in `mcmc.py`. `validation.py`, `preprocessing.py`, `experiment.py`, `cli.py` and `analysis.py` handle their named responsibilities outside the formula path. The core works with arrays, explicit state and keys, without paths or scheduler configuration.

CLI commands are `preprocess`, `prepare` and `run`; both the console entry point and `python -m bayesiancalibration` invoke the same parser. `prepare` freezes observed `theta_s_dagger,F_s,y_tilde,R,s`, fitted `lambda_c` and metadata, excluding evaluation truth. JSON sampler settings use packaged defaults < file < explicitly present CLI flags. `--block-size all` and MALA `--target-accept none` preserve explicit-null override semantics. File and CLI equivalents reach the same factories; run and sampler settings are validated together before sampling. Shared scientific configuration retains profile, CV–NLPD and CV–WMSE selectors and is recorded separately from sampler settings.

`SweepRunner` reuses compiled kernels for a fixed target/method and runs local batches that cannot mix warmup and production. Successful batches produce pickle-free numeric warmup or retained archives. Effective configuration, timings, final tuning and environment are ordinary output records; no state restoration, code-hash gate or migration exists. A failed batch produces focused evidence and no retained archive. Diagnostic mode applies the established GP geometry audit at each completed sweep; failure-only conditional/factor evidence never affects transitions. Adaptation failures distinguish previous valid tuning from actual attempted invalid values. Geometry failures identify the audited completed endpoint. Theta failures explicitly distinguish sweep-start and attempted-final eta; the exact intermediate current/proposal position is unavailable and no purported factor evidence is computed for that position. Candidate coordinate arrays and complete NUTS trajectories remain an acknowledged evidence limitation.

`analysis.py` summarizes every model variable, physical theta, site contrasts, spatial scales/correlations and projected conditional observation means. It computes rank-normalized split/folded R-hat, bulk/tail ESS and mean MCSE using SciPy ranking and BlackJAX's reference diagnostics. Unequal chain lengths use an explicitly reported common prefix. Pooled/per-chain constant flags identify immobility; undefined or short-chain diagnostics serialize as null. The minimum of two chains/eight draws only enables diagnostic calculation, not a convergence claim. No new predictive-noise draws are generated.

Dependency support is now explicit: Python >=3.13 and BlackJAX ==1.6.2, matching the validated adaptation API; the lock removes unused historical resolver branches and transitive dependencies. The tested environment is Python 3.13.15, JAX/JAXlib 0.11.2, NumPy 2.5.3, SciPy 1.18.1, CPU/float64. This is a supported current environment, not an old-run compatibility requirement.

Baseline evidence: an isolated copy of the original code/tests ran 152 tests; 151 passed and the sole failure was the removed shell test's expected Slurm array range (`0-19`) versus its existing print output (`0-11`). No mathematical baseline test failed.

Final local validation (2026-09-30): `python -m unittest discover -s tests -v` ran 163 tests in 374.063 seconds, with 160 passed and three optional supplied-data tests skipped. Those three were then run with `BAYESIANCALIBRATION_TEST_DATA`, together with four chain-analysis tests, and all seven passed (10.886 seconds). Following independent review, pooled/per-chain immobility flags and honest theta/tuning context were verified with focused tests: the final 17 hotpath tests passed (43.022 seconds), the final audited-endpoint label test passed (2.849 seconds), and the two all-method fresh CLI/failure-output regressions passed (11.724 seconds). These focused reruns cover the final diagnostic edits without rerunning unrelated numerical tests.

Installed-wheel execution also regenerated supplied-data preprocessing and a fresh profile-fit prepared archive. A deliberately insufficient two-sweep MH warmup (one chain, seed 20260928) reached the existing zero-scatter freeze gate at sweep 2 for 37 sites. The JSON identified method/chain/warmup/update/quantity/batch and site counts, preserved only the successful first warmup batch, and retained no production draws. This is negative execution/failure-policy evidence for the shortened check, not a posterior or GP-degeneracy diagnosis and not a test failure. All temporary verification artifacts were written outside the repository.

Packaging/inspection: wheel and sdist built offline; installed CLI help, all packaged sampler/scientific JSON defaults and absence of `bayesiancalibration.run`/`comparison` passed. `uv lock --check --offline` resolves 23 packages; Python compilation and `git diff --check` passed. Existing GP source/tests are byte-identical to the starting workspace copy, including the user's CV additions. Historical result files and cluster state were not changed. Next work remains adequate scientific experiment validation, not checkpoint recovery or infrastructure reintroduction.

## 16. Numerical-accuracy cost review (2026-10-07)

Status: review complete; the user subsequently authorized implementation, recorded in Section 17. The user specified that machine-precision agreement is not a project goal. Prefer simpler implementations when numerical differences are negligible for posterior inference and sampler behavior. This supersedes the earlier blanket rationale for retaining every algebraic symmetrization in the pre-benchmark record; it does not resolve the outstanding GP-degeneracy/model questions.

Reviewed the numerical source, host validators, relevant reference/sampler tests, and the primary notes' GP, conditional Gaussian, covariance-adaptation and Fisher-metric formulas. The most defensible simplifications concern repeated symmetry enforcement. No substantial overall sampler cost attributable solely to machine-precision agreement was established: dense covariance factorizations and MMALA derivatives implement the specified mathematics, while redundant averaging adds at most quadratic work beside cubic factorizations.

### Findings and proposed changes

| Priority | Finding | Proposed treatment and consequence |
| --- | --- | --- |
| 1 | `adaptation.update_random_walk_adaptation` averages Welford M2, mirrors its lower triangle, averages the resulting sample covariance again, then mirrors the scaled proposal. Exact `array_equal` symmetry checks in `validation.validate_spd`, `samplers.metropolis._validate_collapsed_inputs` and adaptation initialization help motivate this work. | Retain one M2 normalization and remove the two triangle-mirroring operations and repeated sample-covariance average. Validate symmetry with a practical tolerance, consistent with the existing NUTS check (`rtol=1e-12`, `atol=1e-14`), rather than requiring bitwise equality. Keep the D07 estimator/ridge, exact-zero-scatter rule, invalid-factor checks and production freeze. This reduces code complexity and a small amount of warmup work without changing the covariance estimator in real arithmetic. |
| 2 | Several kernels explicitly average a mathematically symmetric matrix immediately before a standard Gaussian/Cholesky consumer averages it again: `linalg.projected_marginal_moments`, the precision and returned covariance in `gibbs.discrepancy_conditional_moments`/`spatial_mean_conditional_moments`, and the metric/proposal covariance in `samplers.mmala`. Gram products in the IW scale/draw are also averaged. | Remove redundant explicit averages where consumers and returned-moment uses permit. Installed JAX 0.11.2 Cholesky defaults to `symmetrize_input=True`; preserve that standard behavior and its autodiff convention. Review each returned covariance's raw matrix uses before removing its normalization. Small-fixture evidence supports these candidates, but their combined effects have not been validated. No substantial full-sampler speedup is claimed. |
| 3 | Some comparisons between separately evaluated numerical paths require near-machine precision, e.g. eager versus compiled dual averaging in `tests/test_mala_adaptation.py` uses `rtol=1e-14`. Such requirements can constrain otherwise sound simplifications. | Use justified deterministic tolerances for independently computed densities, gradients, moments and tuning; use MCSE and posterior/predictive uncertainty for inference comparisons. Relax a test only when evidence justifies the error budget. Preserve exact checks for PRNG keys, counters, held proposals and frozen production state, which test actual invariants. Tight cheap identity tests need not be weakened merely because they are tight. |

The recommended first change is priority 1 plus consistent tolerant symmetry validation, followed by the local redundant averages in priority 2. Keep the library/field GP conditioning symmetrization and established unjittered singularity audit in this first change: archived field covariances have eigenvalues near rounding scale, where tiny arithmetic differences can affect usability materially. These checks address conditioning and valid Gaussian calculations, rather than demand accurate last digits. Retain float64, stable factorizations/solves, the correction-based coefficient draw, stable transformed log-Jacobians, and the declared regularizers/jitter policy. Do not replace them with explicit inverses, subtractive covariance sampling, clipping, arbitrary tolerances on zero scatter, or a new model remedy.

### Reference probes and timing evidence

Review prototypes ran in separate Python processes from `/private/tmp`; no package source, test, data, archived output, external note or cluster state was modified.

- Removing one explicit averaging boundary at a time on existing bounded/unbounded and loading/dual-branch fixtures changed returned moments/metrics/proposals by at most `2.22e-16` in absolute value. Some outputs were identical. These are small-problem arithmetic checks, not evidence about cumulative sampler or near-singular GP behavior.
- A 600-dimensional collapsed Gaussian likelihood and its covariance gradient agreed exactly when only the explicit projected-covariance average was removed, retaining standard JAX factorization behavior. Synchronized post-compilation timings were approximately 10.4 ms with and 10.5 ms without the redundant average; this establishes no useful speedup.
- For 1,000 Welford updates at 60 sites and three parameters, removing both mirrors and the second covariance average while keeping one M2 average produced identical final proposals on this CPU for both ordinary standardized draws and a `1e6` position-offset stress test. Existing M2 tolerance and NUTS symmetry tolerance passed. Five-group median synchronized timings were 1.64 ms for the current accumulator and 1.31 ms for the prototype per 1,000 updates: about 20% less accumulator time, only about 0.33 microseconds saved per sweep. This excludes the Gibbs/theta updates and does not establish a sampler speedup.
- A more aggressive prototype retaining raw BlackJAX Welford M2 had ordinary-draw proposal asymmetry of approximately `4e-17` relative to the largest proposal entry, which still failed an exact-symmetry check. At the large-offset stress case its asymmetry rose to `4e-12`; retain the single normalization in the first simplification rather than assuming all asymmetry is always negligible.
- Focused existing acceptance tests passed: 13 tests in 9.819 seconds, covering projected Gaussian identities/JIT, Welford/NumPy covariance and D07 regularization, zero/rank-deficient scatter, GP conditioning/jitter/singularity, and independent MMALA conditional/Fisher/forward-reverse proposal references. Command: `PYTHONPATH=src:tests .venv/bin/python -m unittest test_model test_adaptation.CovarianceAdaptationTest test_gp.LibraryConditioningTest test_mmala.MMALATest.test_conditional_moments_covariance_derivatives_and_dense_metric test_mmala.MMALATest.test_sequential_forward_reverse_proposals_match_independent_reference -v`.

The next task at the end of this review was the focused implementation and validation of accumulated effects and ill-conditioned reference cases. Section 17 now records that work. Neither this review nor the subsequent local implementation completes R4's production posterior/predictive comparison.

## 17. Authorized simplification and coefficient kernel extension (2026-10-07)

Status: requested implementation and local acceptance complete. The user authorized the Section 16 work and requested Matérn 3/2 and 5/2 alongside SE, defaulting to Matérn 3/2. Before dependent implementation, the user explicitly selected coefficient K_c only and radial ARD Matérn. Spatial K_theta remains the specified SE with its existing fixed range. This is an explicit model amendment rather than a numerical approximation to the old SE posterior.

Implemented scope: retain one Welford M2 normalization, remove triangle mirroring and repeated sample-covariance normalization, use existing NUTS-scale symmetry tolerances (`rtol=1e-12`, `atol=1e-14`) consistently at covariance boundaries, and remove redundant explicit averages from the Gaussian/Gibbs/MMALA kernels. Keep GP Schur-complement symmetrization/singularity audits, standard JAX Cholesky symmetry/autodiff behavior, float64, all specified regularizers, exact-zero-scatter and fatal-invalid-factor rules, explicit key handling and production freeze. Relax the independent compiled/eager dual-averaging comparison to relative error `1e-10`; other tight mathematical tests and exact invariants remain.

`gp.coefficient_kernel` implements the three unit-amplitude radial correlations. A fixed `kernel` string is passed through profile, CV–NLPD and CV–WMSE objectives/fitting, library factors, field conditioning and unjittered geometry checks; `LibraryGP` and `LengthScaleFit` record it. Matérn uses the analytic polynomial value/derivative extension only at exactly zero squared ARD distance, avoiding undefined sqrt autodiff on Gram diagonals without a distance floor or nugget. The nonzero-distance closed forms are unmodified. `squared_exponential_kernel` remains explicitly SE.

Public input: `library_fit.kernel` in the scientific JSON, or `prepare --kernel {se,matern32,matern52}` as an explicit override. An absent CLI flag preserves a file's choice; an omitted JSON field defaults to `matern32` during fresh preparation. Fitting and target construction use the same frozen kernel and prepared metadata records it explicitly. Loading requires agreement between the fitted-kernel record and scientific settings. A prepared archive missing the kernel must be prepared again; previously fitted SE scales must never silently become Matérn scales. No historical artifact, untracked historical infrastructure, external note or cluster run is changed.

Validation evidence:

- The complete public suite (tracked test modules plus new `test_kernels`) ran 173 tests in 496.460 seconds: 170 passed and three opt-in raw supplied-data preprocessing tests were skipped. No public test failed. This includes independent known-posterior quadrature checks, Gibbs identities, invalid-state/failure reporting, proposal corrections, adaptation/freeze rules, and numerical batching. Untracked historical `test_comparison.py` uses removed checkpoint/ChunkRunner facilities and is outside the public suite; it and the corresponding untracked source files were preserved byte-for-byte.
- New independent kernel references use SciPy distances and the general Bessel Matérn formula. ARD values/diagonals, self/cross/tiny-distance derivatives and coincident-point Hessians, length gradients, library conditioning/density, and profile/CV–NLPD/CV–WMSE scores and gradients pass for all three kernels and both zero/fixed positive jitter. Actual Matérn fits record their selected kernel. Bounded/unbounded loading/dual-branch collapsed gradients and MMALA Fisher matrices match finite-difference/dense references. Existing SE scientific fixtures now request `kernel="se"` explicitly.
- Fresh CLI tests pass for defaults, file preservation, explicit overrides, fitted/scientific metadata agreement and missing-kernel rejection. All five samplers passed preparation, four warmup sweeps and two retained sweeps for each Matérn kernel. These short runs validate execution, not convergence.
- An isolated before/after SE comparison used the same inputs and keys across all five methods, in unbounded dual-branch and bounded loading-only cases: 20 warmup plus 64 retained sweeps for each of ten chains. Discrete outcomes and final PRNG keys agreed. Maximum absolute differences across states, tuning and diagnostics were `6.26e-12` (MH), `1.07e-13` (MALA), `1.50e-12` (MMALA), and zero for both NUTS variants. The maximum scaled error was `2.28e-13`. A Gaussian reference with prior condition number approximately `1e9` had at most `1.11e-16` absolute difference; joint/collapsed densities, conditional gradients and coefficient draws agreed. These checks do not make identical trajectories a requirement for other implementations/backends.
- Fresh public preparation of the existing Stage 14a 20-run/60-site/5-coefficient arrays passed for both Matérn kernels, with zero jitter. Matérn 3/2 selected lambda_c approximately `(23.39840,37.63299,40.53888)`, profile objective `-393.30996`, and two successful starts out of three. Matérn 5/2 selected approximately `(9.16728,15.31183,16.81948)`, objective `-385.64007`, and three successful starts. Each prepared target passed four independently generated complete initial-state validations. The unsuccessful Matérn 3/2 start follows the existing recorded-attempt policy; no repair, retry or optimizer-policy change was added. These are library fits/initialization evidence, not calibration posterior comparisons. Temporary prepared files were removed; historical artifacts remain unchanged.
- Python compilation, CLI help and `git diff --check` passed. No dependency was added. Review/comparison logs and probes were written under `/private/tmp`; no cluster run was launched.

The requested implementation is complete. No kernel-selection question remains open. Next scientific work remains adequate fresh posterior/predictive validation and diagnosis of full-protocol numerical failures; this task does not establish long-chain inference agreement or resolve the archived GP-degeneracy questions.
