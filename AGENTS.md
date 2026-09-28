# Coding-agent instructions

- Keep `docs/implementation-plan.md` as the single living planning document. Update its decisions, status, next task, and validation evidence as work proceeds; do not create competing implementation plans.
- Treat the statistical notes referenced in the implementation plan as the primary method specifications. Do not replace the model, target distribution, or sampler with a generic calibration implementation. If the notes and implementation plan conflict, report the conflict rather than silently choosing one.
- Model-specific code should follow the source notes' notation whenever practical: prefer `c_f`, `m_theta`, and other matching symbols over generic names such as `coefficients` or `coefficient_mean`. Use readable ASCII equivalents, document mathematical meanings and array shapes, and preserve distinctions between different quantities; do not mechanically substitute symbols with different meanings. Generic numerical utilities may retain generic names such as `mean` and `covariance`.
- Do not silently resolve open statistical or model-defining questions. Record proposed resolutions and their consequences in the implementation plan, and obtain an explicit decision before implementing behavior that depends on them. Independent tests, reference calculations, and validation work that do not depend on the unresolved choice may proceed.
- Prioritize, in order:
  1. statistical correctness
  2. numerical correctness
  3. testability
  4. clarity
  5. reproducibility
  6. performance
- Use transparent Python/JAX research code with float64 numerical calculations, explicit PRNG keys, documented array shapes, and a single documented stacking convention. Prefer pure JAX numerical kernels where practical, while keeping orchestration simple and readable. Keep model state, adaptation state, derived caches, and diagnostics conceptually distinct.
- Prefer existing JAX, SciPy, or NumPy functions over new helper functions unless there is a specific reason to write one. Implement a helper only when the available functions cannot meet the requirement, or when it has a clear performance, numerical-stability, autodiff, JIT, or readability benefit. Briefly document the reason for each custom helper.
- Use numerically stable factorizations and linear solves rather than explicit matrix inverses or determinants unless there is a documented reason otherwise. Treat nuggets, jitter, regularization, truncation, and other approximations as explicit model or numerical choices, and apply them consistently across compared methods.
- Validate mathematical identities and small reference problems before optimization. Do not mark a milestone complete until its stated acceptance tests pass. Do not use stale cached quantities after their dependencies change.
- Implement only the requested milestone or task. Preserve unrelated user edits. Avoid unrelated refactors, abstractions, or new dependencies unless there is a concrete need.
- After completing work, report what changed, what validation was performed, what passed or failed, and any remaining limitations or unresolved questions. Update `docs/implementation-plan.md` accordingly.
