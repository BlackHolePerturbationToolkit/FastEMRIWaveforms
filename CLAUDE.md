# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working
with code in this repository.

**LISA Analysis Tools–wide conventions:** [`../LISAanalysistools/docs/conventions.md`](../LISAanalysistools/docs/conventions.md) (canonical). **Cross-repo architecture:** [`../LISAanalysistools/docs/architecture-map.md`](../LISAanalysistools/docs/architecture-map.md). FEW has no `docs/codebase-map.md` yet; the batched / pooled trajectory work is described in [`docs/traj-batching.md`](docs/traj-batching.md).

## Role and current state (2026-10, branch `gpu_backend`)

FEW (`few`) is the EMRI waveform package of the LISA Analysis Tools stack.

- **Backends** come from `gpubackendtools` (GBT): `import few` registers
  `few_cpu`, `few_cuda11x`, `few_cuda12x`, `few_cuda13x`
  (`src/few/cutils/__init__.py`), and every backend-aware class derives from
  `few.utils.baseclasses.FewBackendConsumer` (`force_backend=`). FEW has no
  package-level `get_backend` / `has_backend`; use GBT's with `few_*` names.
  `gpubackendtools` is imported by `few` but is missing from the
  `pyproject.toml` dependencies. The `enabled_backends` config option is inert.
- **Native code:** Cython wrappers (`*.pyx`) over `.cu` kernels in
  `src/few/cutils/`; the CPU backend compiles a `.cxx` copy of each `.cu`.
  No JAX code. Constants come from the committed `lisaconstants_values.h`
  (`lisaconstants==2.0.2`; see the LAT conventions). Trajectories (DOP853 +
  numba) are CPU-only and use no backend.
- **Production reach (9mo/12mo global-fit runs, through LAT
  `lisatools.sources.emri`):** only `FastKerrEccentricEquatorialFlux`, built
  inside `GenerateEMRIWaveform` on the run's GPU backend (`cuda13x` on the
  cluster) and called with `return_sparse_holder=True` (trajectory +
  `AmpInterpKerrEccEq` amplitudes + Ylm + mode selection, no mode sum; returns
  `few.waveform.base.SparseInfoHolder`), plus `few.trajectory.pool`
  (`TrajectoryCache` / `TrajectoryPool`) for the trajectory fan-out. The full
  mode sum runs only on LAT's cross-check / fallback path.
- **Known defects (report-only, not fixed):** importing `few` reseeds NumPy's
  global RNG (`np.random.seed(5)` at the top of `trajectory/dopr853.py`);
  `FDInterpolatedModeSum.sum` and `DOPR853.controllerSuccess` contain
  `breakpoint()` calls; `few.utils.globals.__all__` lists undefined names;
  `Pn5AAK.sanity_check_init` calls an un-imported `get_logger` (NameError for
  AAK calls with `m2/m1 > 1e-4` or `a < 0`). Dead or broken helpers carry a
  `# TODO check:` comment and are excluded from the API pages.

## Backend implementation hierarchy (sprint-wide rule)

When implementing or modifying an algorithm that exists across multiple
backends (GPU C++ / CPU C++ / JAX), follow this hierarchy:

1. **GPU C++ (CUDA) leads.** This is the canonical performance target
   and reference implementation. New algorithms and optimizations are
   designed for the GPU first; CPU and JAX paths follow.

2. **CPU C++ mirrors GPU C++ as closely as possible.** Same kernel
   structure, same algorithm, same data flow — use `#ifdef __CUDACC__`
   or shared compile-time macros (in FEW's `src/few/cutils/global.h`:
   `CUDA_CALLABLE_MEMBER`, `CUDA_KERNEL`, `CUDA_SHARED`,
   `CUDA_SYNC_THREADS`) to bridge platform differences. The CPU path
   exists primarily for testing and CPU-only environments; it must
   not diverge in algorithm or output beyond floating-point order of
   operations.

3. **CPU C++ must reproduce the overall lisatools computation.**
   Against the lisatools reference (e.g. `FDSignal.transform`,
   `TDSignal.transform`, `XYZ2SensitivityMatrix`), match to machine
   precision (≤ 1e-15 mismatch) in direct modes; cache/approximation
   modes have documented per-feature error budgets.

4. **JAX may diverge internally** — design it to be JAX-efficient.
   JAX-CPU and JAX-GPU compilation targets may even differ. Use
   JAX-native idioms (`jax.lax.scan`, `jax.vmap`, static-shape
   `dynamic_slice` + masks, functional carries) rather than
   mechanically translating CUDA shared memory / register caches.

5. **JAX must match C++ inner-product outputs.** End-to-end
   likelihood quantities (`<d|h>`, `<h|h>`, swap_ll 5 terms) must
   match the C++ to floating-point precision (reldiff ≲ 1e-12) on
   representative test cases. Intermediate quantities (raw templates,
   per-chunk WDM coefficients) may differ at FP precision due to
   summation order — validate at the inner-product level.

**Workflow for a new feature.** GPU C++ → CPU C++ via `#ifdef` → JAX
with JAX-native idioms → cross-backend inner-product validation.
